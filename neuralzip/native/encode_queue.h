nb::list compress_mixed_streams_many(
    nb::handle tensor_source,
    Py_ssize_t min_saving_bytes,
    int requested_threads,
    double compression_threshold) {
    require<nb::value_error>(
        min_saving_bytes >= 0 && requested_threads >= 0,
        "min_saving_bytes and threads must be non-negative.");
    require<nb::value_error>(
        compression_threshold > 0.0 && compression_threshold <= 1.0,
        "compression_threshold must be in (0, 1].");

    nb::tuple tensor_sequence(tensor_source);
    struct MixedTensorRequest {
        ReadOnlyBuffer raw;
        int element_size = 0;
        bool bit_reorder = false;
        std::vector<MixedStreamSpec> specs;
        std::vector<std::size_t> group_sizes;
        std::size_t result_offset = 0;
    };
    struct MixedModelJob {
        std::size_t tensor_index;
        int stream;
        std::size_t start;
        std::size_t length;
    };

    std::vector<MixedTensorRequest> requests(tensor_sequence.size());
    std::vector<MixedModelJob> jobs;
    std::size_t total_input_bytes = 0;
    for (std::size_t tensor_index = 0;
         tensor_index < tensor_sequence.size();
         ++tensor_index) {
        nb::tuple item(nb::handle(tensor_sequence[tensor_index].ptr()));
        require<nb::type_error>(
            item.size() == 5,
            "Each tensor spec must contain data, element_size, bit_reorder, "
            "chunk_size, and stream_modes.");
        nb::handle raw_source(item[0].ptr());
        const int element_size =
            nb::cast<int>(nb::handle(item[1].ptr()));
        const bool bit_reorder = nb::cast<bool>(nb::handle(item[2].ptr()));
        const Py_ssize_t requested_chunk_size =
            nb::cast<Py_ssize_t>(nb::handle(item[3].ptr()));
        nb::handle mode_source(item[4].ptr());

        require<nb::value_error>(
            element_size == 2 || element_size == 4,
            "element_size must be 2 or 4.");
        require<nb::value_error>(
            requested_chunk_size > 0,
            "chunk_size must be positive.");

        MixedTensorRequest& request = requests[tensor_index];
        request.raw.acquire(raw_source);
        require<nb::value_error>(
            request.raw.size() % element_size == 0,
            "data length must be divisible by element_size.");
        const std::size_t raw_bytes =
            static_cast<std::size_t>(request.raw.size());
        require<std::overflow_error>(
            raw_bytes <= std::numeric_limits<std::size_t>::max()
                - total_input_bytes,
            "Total mixed input size overflowed.");
        total_input_bytes += raw_bytes;
        request.element_size = element_size;
        request.bit_reorder = bit_reorder;
        request.specs = parse_mixed_stream_specs(
            mode_source,
            element_size,
            static_cast<std::size_t>(requested_chunk_size));
        request.group_sizes.assign(
            static_cast<std::size_t>(element_size), 0);
        request.result_offset = jobs.size();

        const std::size_t symbol_count =
            raw_bytes / static_cast<std::size_t>(element_size);
        for (int stream = 0; stream < element_size; ++stream) {
            const std::size_t step =
                request.specs[static_cast<std::size_t>(stream)].chunk_size;
            for (std::size_t start = 0;
                 start < symbol_count;
                 start += step) {
                jobs.push_back({
                    tensor_index,
                    stream,
                    start,
                    std::min(step, symbol_count - start),
                });
                request.group_sizes[
                    static_cast<std::size_t>(stream)] += 1;
            }
        }
    }

    std::vector<MixedBlockResult> results(jobs.size());
    {
        nb::gil_scoped_release release;
        run_parallel_jobs(
            jobs.size(),
            total_input_bytes,
            requested_threads,
            [&](std::size_t job_index) {
                const MixedModelJob& job = jobs[job_index];
                const MixedTensorRequest& request =
                    requests[job.tensor_index];
                const MixedStreamSpec& spec = request.specs[
                    static_cast<std::size_t>(job.stream)];
                MixedBlockResult& result = results[job_index];
                try {
                    process_mixed_stream_block(
                        result,
                        spec,
                        request.raw.data(),
                        request.element_size,
                        request.bit_reorder,
                        job.stream,
                        job.start,
                        job.length,
                        min_saving_bytes,
                        compression_threshold);
                } catch (const std::exception& error) {
                    result.error = error.what();
                } catch (...) {
                    result.error =
                        "Unknown model-wide mixed encoding failure.";
                }
            });
    }

    for (const auto& result : results) {
        if (!result.error.empty()) {
            throw nb::value_error(result.error.c_str());
        }
    }

    nb::list output;
    for (const auto& request : requests) {
        output.append(mixed_groups_to_python(
            request.element_size,
            request.group_sizes,
            results,
            request.result_offset));
    }
    return output;
}
