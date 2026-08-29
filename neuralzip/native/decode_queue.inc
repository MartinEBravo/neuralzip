nb::list decompress_mixed_streams_many(
    nb::handle tensor_source,
    int requested_threads) {
    require<nb::value_error>(
        requested_threads >= 0,
        "threads must be non-negative.");

    nb::tuple tensor_sequence(tensor_source);
    const std::size_t tensor_count = tensor_sequence.size();
    std::vector<MixedStreamDecodeTensor> requests(tensor_count);
    std::vector<MixedStreamDecodeChunk> chunks;
    std::vector<MixedStreamAlignedDecodeJob> aligned_jobs;
    std::vector<std::size_t> fallback_chunk_jobs;
    std::vector<MixedStreamCombineJob> combine_jobs;

    std::size_t total_output_bytes = 0;
    for (std::size_t tensor_index = 0;
         tensor_index < tensor_count;
         ++tensor_index) {
        nb::tuple item(nb::handle(tensor_sequence[tensor_index].ptr()));
        require<nb::type_error>(
            item.size() == 4,
            "Each decode spec must contain element_size, bit_reorder, numel, "
            "and groups.");
        const int element_size =
            nb::cast<int>(nb::handle(item[0].ptr()));
        const bool bit_reorder = nb::cast<bool>(nb::handle(item[1].ptr()));
        const Py_ssize_t requested_numel =
            nb::cast<Py_ssize_t>(nb::handle(item[2].ptr()));
        nb::handle groups_source(item[3].ptr());

        require<nb::value_error>(
            element_size == 2 || element_size == 4,
            "element_size must be 2 or 4.");
        require<nb::value_error>(
            requested_numel >= 0,
            "numel must be non-negative.");

        const std::size_t numel =
            static_cast<std::size_t>(requested_numel);
        const std::size_t width =
            static_cast<std::size_t>(element_size);
        require<std::overflow_error>(
            numel <= static_cast<std::size_t>(PY_SSIZE_T_MAX) / width,
            "Decoded tensor byte size exceeds Py_ssize_t.");
        const std::size_t raw_bytes = numel * width;
        require<std::overflow_error>(
            raw_bytes <= std::numeric_limits<std::size_t>::max()
                - total_output_bytes,
            "Total decoded tensor size overflowed.");
        total_output_bytes += raw_bytes;

        MixedStreamDecodeTensor& request = requests[tensor_index];
        request.element_size = element_size;
        request.bit_reorder = bit_reorder;
        nb::bytearray output(
            static_cast<const void*>(nullptr), raw_bytes);
        request.output_data =
            static_cast<std::uint8_t*>(output.data());
        request.output = std::move(output);

        nb::tuple group_sequence(groups_source);
        require<nb::value_error>(
            group_sequence.size() == static_cast<std::size_t>(element_size),
            "groups must contain exactly one group per byte stream.");

        std::array<std::vector<std::size_t>, 4>
            stream_chunk_indices;
        for (int stream = 0; stream < element_size; ++stream) {
            nb::tuple chunk_sequence(
                nb::handle(group_sequence[stream].ptr()));
            std::size_t output_offset = 0;
            for (std::size_t chunk_index = 0;
                 chunk_index < chunk_sequence.size();
                 ++chunk_index) {
                nb::tuple chunk_source(
                    nb::handle(chunk_sequence[chunk_index].ptr()));
                require<nb::type_error>(
                    chunk_source.size() == 8,
                    "Each chunk must contain mode, payload, symbol_count, "
                    "decodebook, vocabulary, ngram_n, escapes, and tail.");
                const std::string kind = nb::cast<std::string>(
                    nb::handle(chunk_source[0].ptr()));
                nb::handle payload_source(chunk_source[1].ptr());
                const Py_ssize_t requested_symbols =
                    nb::cast<Py_ssize_t>(
                        nb::handle(chunk_source[2].ptr()));
                nb::handle decodebook_source(chunk_source[3].ptr());
                nb::handle vocab_source(chunk_source[4].ptr());
                const int ngram_n = nb::cast<int>(
                    nb::handle(chunk_source[5].ptr()));
                nb::handle escape_source(chunk_source[6].ptr());
                nb::handle tail_source(chunk_source[7].ptr());

                require<nb::value_error>(
                    requested_symbols >= 0,
                    "chunk symbol_count must be non-negative.");
                const std::size_t symbol_count =
                    static_cast<std::size_t>(requested_symbols);
                require<nb::value_error>(
                    symbol_count <= numel - output_offset,
                    "chunk symbol counts exceed the tensor numel.");

                chunks.emplace_back();
                MixedStreamDecodeChunk& chunk = chunks.back();
                chunk.tensor_index = tensor_index;
                chunk.stream = stream;
                chunk.output_offset = output_offset;
                chunk.symbol_count = symbol_count;
                chunk.payload.acquire(payload_source);

                if (kind == "raw") {
                    chunk.mode = MixedMode::Raw;
                    require<nb::value_error>(
                        chunk.payload.size() == requested_symbols,
                        "raw chunk payload size must equal symbol_count.");
                } else if (kind == "shared_huff0") {
                    chunk.mode = MixedMode::SharedHuff0;
                    require<nb::value_error>(
                        !decodebook_source.is_none(),
                        "shared_huff0 chunks require a decodebook handle.");
                    require<nb::value_error>(
                        symbol_count <= kMaxBlockSize,
                        "shared_huff0 chunk exceeds the Huff0 block limit.");
                    chunk.decodebook_owner =
                        nb::borrow<nb::object>(decodebook_source);
                    chunk.decodebook = capsule_data<Huff0Decodebook>(
                        decodebook_source, "neuralzip.Huff0Decodebook");
                } else if (kind == "packed_huff0") {
                    require<nb::value_error>(
                        !decodebook_source.is_none(),
                        "packed_huff0 chunks require a decodebook handle.");
                    require<nb::value_error>(
                        ngram_n >= 2 && ngram_n <= 8,
                        "packed_huff0 requires ngram_n in [2, 8].");
                    ReadOnlyBuffer vocab_bytes(vocab_source);
                    chunk.vocab = vocab_from_bytes(
                        vocab_bytes.data(), vocab_bytes.size(), ngram_n);
                    chunk.escapes.acquire(escape_source);
                    chunk.tail.acquire(tail_source);
                    const std::size_t token_count = symbol_count
                        / static_cast<std::size_t>(ngram_n);
                    require<nb::value_error>(
                        token_count <= kMaxBlockSize,
                        "packed_huff0 chunk exceeds the Huff0 block limit.");
                    chunk.mode = MixedMode::PackedHuff0;
                    chunk.ngram_n = ngram_n;
                    chunk.decodebook_owner =
                        nb::borrow<nb::object>(decodebook_source);
                    chunk.decodebook = capsule_data<Huff0Decodebook>(
                        decodebook_source, "neuralzip.Huff0Decodebook");
                } else {
                    throw nb::value_error(
                        "Unsupported mixed-stream decode chunk mode.");
                }

                stream_chunk_indices[
                    static_cast<std::size_t>(stream)].push_back(
                        chunks.size() - 1U);
                output_offset += symbol_count;
            }
            require<nb::value_error>(
                output_offset == numel,
                "Each decoded byte stream must contain exactly numel symbols.");
        }

        bool chunks_are_aligned = true;
        const std::size_t aligned_chunk_count =
            stream_chunk_indices[0].size();
        for (std::size_t stream = 1; stream < width; ++stream) {
            if (stream_chunk_indices[stream].size()
                != aligned_chunk_count) {
                chunks_are_aligned = false;
                break;
            }
        }
        if (chunks_are_aligned) {
            for (std::size_t chunk_index = 0;
                 chunk_index < aligned_chunk_count;
                 ++chunk_index) {
                const MixedStreamDecodeChunk& first = chunks[
                    stream_chunk_indices[0][chunk_index]];
                for (std::size_t stream = 1;
                     stream < width;
                     ++stream) {
                    const MixedStreamDecodeChunk& other = chunks[
                        stream_chunk_indices[stream][chunk_index]];
                    if (other.output_offset != first.output_offset
                        || other.symbol_count != first.symbol_count) {
                        chunks_are_aligned = false;
                        break;
                    }
                }
                if (!chunks_are_aligned) {
                    break;
                }
            }
        }

        if (chunks_are_aligned) {
            for (std::size_t chunk_index = 0;
                 chunk_index < aligned_chunk_count;
                 ++chunk_index) {
                MixedStreamAlignedDecodeJob job;
                const MixedStreamDecodeChunk& first = chunks[
                    stream_chunk_indices[0][chunk_index]];
                job.tensor_index = tensor_index;
                job.output_offset = first.output_offset;
                job.symbol_count = first.symbol_count;
                for (std::size_t stream = 0;
                     stream < width;
                     ++stream) {
                    job.chunk_indices[stream] =
                        stream_chunk_indices[stream][chunk_index];
                }
                aligned_jobs.push_back(std::move(job));
            }
        } else {
            request.lanes.resize(width);
            for (auto& lane : request.lanes) {
                lane.resize(numel);
            }
            for (std::size_t stream = 0;
                 stream < width;
                 ++stream) {
                fallback_chunk_jobs.insert(
                    fallback_chunk_jobs.end(),
                    stream_chunk_indices[stream].begin(),
                    stream_chunk_indices[stream].end());
            }

            constexpr std::size_t kCombineJobBytes = 256U << 10;
            const std::size_t combine_step = std::max<std::size_t>(
                1U, kCombineJobBytes / width);
            for (std::size_t start = 0;
                 start < numel;
                 start += combine_step) {
                combine_jobs.push_back({
                    tensor_index,
                    start,
                    std::min(combine_step, numel - start),
                });
            }
        }
    }

    std::string native_error;
    const std::string* decode_error = nullptr;
    {
        nb::gil_scoped_release release;
        try {
            const std::size_t aligned_job_count = aligned_jobs.size();
            const std::size_t first_phase_job_count =
                aligned_job_count + fallback_chunk_jobs.size();
            run_parallel_jobs(
                first_phase_job_count,
                total_output_bytes,
                requested_threads,
                [&](std::size_t job_index) {
                    if (job_index < aligned_job_count) {
                        MixedStreamAlignedDecodeJob& job =
                            aligned_jobs[job_index];
                        const MixedStreamDecodeTensor& request =
                            requests[job.tensor_index];
                        try {
                            const std::size_t width =
                                static_cast<std::size_t>(
                                    request.element_size);
                            std::size_t compressed_lane_count = 0;
                            for (std::size_t stream = 0;
                                 stream < width;
                                 ++stream) {
                                const MixedMode mode =
                                    chunks[job.chunk_indices[stream]].mode;
                                if (mode != MixedMode::Raw) {
                                    ++compressed_lane_count;
                                }
                            }
                            require<std::length_error>(
                                compressed_lane_count == 0
                                    || job.symbol_count
                                        <= std::numeric_limits<std::size_t>::max()
                                            / compressed_lane_count,
                                "Aligned decode scratch size overflowed.");

                            thread_local std::vector<std::uint8_t> scratch;
                            scratch.resize(
                                compressed_lane_count * job.symbol_count);
                            std::array<const std::uint8_t*, 4> lane_data{};
                            std::size_t scratch_offset = 0;
                            for (std::size_t stream = 0;
                                 stream < width;
                                 ++stream) {
                                const MixedStreamDecodeChunk& chunk = chunks[
                                    job.chunk_indices[stream]];
                                if (chunk.mode == MixedMode::Raw) {
                                    lane_data[stream] = chunk.payload.data();
                                    continue;
                                }
                                std::uint8_t* destination = nullptr;
                                if (chunk.symbol_count > 0) {
                                    destination =
                                        scratch.data() + scratch_offset;
                                }
                                decode_mixed_stream_chunk_into(
                                    chunk, destination);
                                lane_data[stream] = destination;
                                scratch_offset += chunk.symbol_count;
                            }
                            restore_mixed_stream_chunk(
                                request,
                                lane_data,
                                job.output_offset,
                                job.symbol_count);
                        } catch (const std::exception& error) {
                            job.error = error.what();
                        } catch (...) {
                            job.error =
                                "Unknown aligned mixed-stream decode failure.";
                        }
                        return;
                    }

                    MixedStreamDecodeChunk& chunk = chunks[
                        fallback_chunk_jobs[
                            job_index - aligned_job_count]];
                    MixedStreamDecodeTensor& request =
                        requests[chunk.tensor_index];
                    std::uint8_t* destination = nullptr;
                    if (chunk.symbol_count > 0) {
                        destination = request.lanes[
                            static_cast<std::size_t>(chunk.stream)].data()
                            + chunk.output_offset;
                    }
                    try {
                        decode_mixed_stream_chunk_into(chunk, destination);
                    } catch (const std::exception& error) {
                        chunk.error = error.what();
                    } catch (...) {
                        chunk.error =
                            "Unknown mixed-stream decode failure.";
                    }
                });

            for (const auto& job : aligned_jobs) {
                if (!job.error.empty()) {
                    decode_error = &job.error;
                    break;
                }
            }
            if (decode_error == nullptr) {
                for (const auto& chunk : chunks) {
                    if (!chunk.error.empty()) {
                        decode_error = &chunk.error;
                        break;
                    }
                }
            }
            if (decode_error == nullptr) {
                run_parallel_jobs(
                    combine_jobs.size(),
                    total_output_bytes,
                    requested_threads,
                    [&](std::size_t job_index) {
                        const MixedStreamCombineJob& job =
                            combine_jobs[job_index];
                        restore_mixed_stream_tensor_range(
                            requests[job.tensor_index],
                            job.start,
                            job.length);
                    });
            }
        } catch (const std::exception& error) {
            native_error = error.what();
        } catch (...) {
            native_error =
                "Unknown model-wide mixed-stream decode failure.";
        }
    }

    if (!native_error.empty()) {
        throw std::runtime_error(native_error);
    }
    if (decode_error != nullptr) {
        throw nb::value_error(decode_error->c_str());
    }

    nb::list output;
    for (const auto& request : requests) {
        output.append(request.output);
    }
    return output;
}
