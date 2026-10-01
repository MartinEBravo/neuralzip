nb::list decompress_mixed_streams_many(nb::handle tensor_source, int requested_threads) {
    require<nb::value_error>(requested_threads >= 0, "threads must be non-negative.");
    nb::tuple tensor_sequence(tensor_source);
    std::vector<MixedStreamDecodeTensor> requests(tensor_sequence.size());
    std::vector<MixedStreamDecodeChunk> chunks;
    std::vector<MixedStreamAlignedDecodeJob> jobs;
    std::size_t total_output_bytes = 0;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        nb::tuple item(nb::handle(tensor_sequence[i].ptr()));
        require<nb::type_error>(item.size() == 4, "Decode spec needs width, reorder, numel, and groups.");
        auto& request = requests[i];
        request.element_size = nb::cast<int>(nb::handle(item[0].ptr()));
        request.bit_reorder = nb::cast<bool>(nb::handle(item[1].ptr()));
        const auto count = nb::cast<Py_ssize_t>(nb::handle(item[2].ptr()));
        require<nb::value_error>((request.element_size == 2 || request.element_size == 4) && count >= 0,
            "Invalid decoded width or element count.");
        const auto width = static_cast<std::size_t>(request.element_size);
        const auto numel = static_cast<std::size_t>(count);
        require<std::overflow_error>(numel <= static_cast<std::size_t>(PY_SSIZE_T_MAX) / width,
            "Decoded tensor exceeds Python's buffer limit.");
        const auto raw_size = numel * width;
        require<std::overflow_error>(raw_size <= std::numeric_limits<std::size_t>::max() - total_output_bytes,
            "Total decoded tensor size overflowed.");
        total_output_bytes += raw_size;
        nb::bytearray output(static_cast<const void*>(nullptr), raw_size);
        request.output_data = static_cast<std::uint8_t*>(output.data());
        request.output = std::move(output);
        nb::tuple groups(nb::handle(item[3].ptr()));
        require<nb::value_error>(groups.size() == width, "Decode needs one group per byte stream.");
        std::array<std::vector<std::size_t>, 4> indices;
        for (std::size_t stream = 0; stream < width; ++stream) {
            std::size_t offset = 0;
            for (auto source : nb::tuple(nb::handle(groups[stream].ptr()))) {
                nb::tuple record(source);
                require<nb::type_error>(record.size() == 4, "Chunk needs mode, payload, count, and decode table.");
                const auto kind = nb::cast<std::string>(nb::handle(record[0].ptr()));
                const auto length = nb::cast<Py_ssize_t>(nb::handle(record[2].ptr()));
                require<nb::value_error>(length >= 0 && static_cast<std::size_t>(length) <= numel - offset,
                    "Chunk exceeds its tensor element count.");
                chunks.emplace_back();
                auto& chunk = chunks.back();
                chunk.symbol_count = static_cast<std::size_t>(length);
                chunk.output_offset = offset;
                chunk.payload.acquire(nb::handle(record[1].ptr()));
                if (kind == "raw") {
                    require<nb::value_error>(chunk.payload.size() == length, "Raw payload size mismatch.");
                } else {
                    require<nb::value_error>(kind == "shared_huff0" || kind == "mixed_huffman",
                        "Unsupported decoded chunk mode.");
                    nb::handle handle(record[3].ptr());
                    require<nb::value_error>(!handle.is_none(), "Compressed chunk needs a decode table.");
                    chunk.decodebook_owner = nb::borrow<nb::object>(handle);
                    chunk.decodebook = capsule_data<Huff0Decodebook>(handle, "neuralzip.Huff0Decodebook");
                    chunk.mode = kind == "mixed_huffman" ? MixedMode::MixedHuffman : MixedMode::SharedHuff0;
                    require<nb::value_error>(static_cast<bool>(chunk.decodebook->mixed) == (kind == "mixed_huffman"),
                        "Decode table format does not match chunk mode.");
                    require<nb::value_error>(chunk.mode != MixedMode::SharedHuff0 || chunk.symbol_count <= kMaxBlockSize,
                        "Shared chunk exceeds Huff0's block limit.");
                }
                indices[stream].push_back(chunks.size() - 1);
                offset += chunk.symbol_count;
            }
            require<nb::value_error>(offset == numel, "Byte stream does not reconstruct exactly numel values.");
        }
        for (std::size_t stream = 1; stream < width; ++stream)
            require<nb::value_error>(indices[stream].size() == indices[0].size(), "Unaligned chunk groups.");
        for (std::size_t j = 0; j < indices[0].size(); ++j) {
            const auto& first = chunks[indices[0][j]];
            MixedStreamAlignedDecodeJob job;
            job.tensor_index = i;
            job.output_offset = first.output_offset;
            job.symbol_count = first.symbol_count;
            for (std::size_t stream = 0; stream < width; ++stream) {
                const auto index = indices[stream][j];
                require<nb::value_error>(chunks[index].output_offset == job.output_offset
                    && chunks[index].symbol_count == job.symbol_count, "Unaligned chunk boundaries.");
                job.chunk_indices[stream] = index;
            }
            jobs.push_back(std::move(job));
        }
    }
    {
        nb::gil_scoped_release release;
        run_parallel_jobs(jobs.size(), total_output_bytes, requested_threads, [&](std::size_t i) {
            auto& job = jobs[i];
            const auto& request = requests[job.tensor_index];
            try {
                thread_local std::vector<std::uint8_t> scratch;
                scratch.resize(job.symbol_count * request.element_size);
                std::array<const std::uint8_t*, 4> lanes{};
                for (int stream = 0; stream < request.element_size; ++stream) {
                    const auto& chunk = chunks[job.chunk_indices[stream]];
                    if (chunk.mode == MixedMode::Raw) { lanes[stream] = chunk.payload.data(); continue; }
                    auto* destination = scratch.data() + stream * job.symbol_count;
                    if (chunk.mode == MixedMode::MixedHuffman)
                        decode_mixed_block_into(chunk.payload.data(), chunk.payload.size(), *chunk.decodebook->mixed,
                            destination, chunk.symbol_count);
                    else
                        decode_huff0_block_into(chunk.payload.data(), chunk.payload.size(), destination,
                            chunk.symbol_count, *chunk.decodebook);
                    lanes[stream] = destination;
                }
                restore_mixed_stream_chunk(request, lanes, job.output_offset, job.symbol_count);
            } catch (const std::exception& error) { job.error = error.what(); }
            catch (...) { job.error = "Unknown native decoding failure."; }
        });
    }
    for (const auto& job : jobs)
        if (!job.error.empty()) throw nb::value_error(job.error.c_str());
    nb::list output;
    for (const auto& request : requests) output.append(request.output);
    return output;
}
