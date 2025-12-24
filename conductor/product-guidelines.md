# Product Guidelines

## Tone and Voice
*   **Primary Style: Pragmatic and Direct.** Documentation and reports must be efficient, focusing on immediate utility ("how-to") and results without unnecessary filler.
*   **Secondary Style: Technical and Precise.** When addressing performance-focused users, provide deep engineering details, data accuracy, and technical specifics.

## Presentation and Reporting
*   **Detailed Metric Breakdowns:** Benchmark results must include comprehensive data such as standard deviation, peak memory usage, and granular hardware utilization metrics to support deep technical analysis.
*   **Functional Aesthetic:** Visual reports and charts should prioritize high-contrast and functional design, ensuring maximum readability and clarity within markdown and CLI environments.

## User Interaction and Messaging
*   **Verbosity Control:** The system must implement configurable output levels (e.g., silent, normal, debug), allowing users to control the level of detail provided during execution.
*   **Fail-Fast Validation:** Before initiating intensive tasks like benchmarks, the platform must proactively validate all required external dependencies, model files, and hardware compatibility to prevent late-stage failures.
