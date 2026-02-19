package com.loadbalancer.metrics;

import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;
import java.util.List;

/**
 * Collects and computes statistics for a single simulation run.
 *
 * Tracks per-request latencies and provides aggregate metrics:
 *   - Mean, P50, P95, P99 latencies
 *   - Throughput (requests per unit time)
 *   - Regret (cumulative difference from optimal)
 */
public class MetricsCollector {

    private final String algorithmName;
    private final List<Double> latencies;
    private final List<Integer> serverSelections;

    // Cumulative regret tracking
    private double optimalLatency;     // best achievable latency (set from simulation)
    private double cumulativeRegret;

    // Moving window for rolling average (window of last N requests)
    private final int windowSize;
    private final List<Double> rollingWindow;

    public MetricsCollector(String algorithmName, double optimalLatency, int windowSize) {
        this.algorithmName = algorithmName;
        this.latencies = new ArrayList<>();
        this.serverSelections = new ArrayList<>();
        this.optimalLatency = optimalLatency;
        this.cumulativeRegret = 0.0;
        this.windowSize = windowSize;
        this.rollingWindow = new ArrayList<>();
    }

    /**
     * Records a completed request's result.
     */
    public void record(int serverIndex, double latency) {
        latencies.add(latency);
        serverSelections.add(serverIndex);

        // Regret = difference from optimal latency
        double regret = latency - optimalLatency;
        if (regret > 0) {
            cumulativeRegret += regret;
        }

        // Update rolling window
        rollingWindow.add(latency);
        if (rollingWindow.size() > windowSize) {
            rollingWindow.remove(0);
        }
    }

    /**
     * Returns average latency over all recorded requests.
     */
    public double getMeanLatency() {
        return latencies.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    /**
     * Returns rolling average of the last N requests.
     */
    public double getRollingAverage() {
        return rollingWindow.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    /**
     * Returns the Nth percentile latency.
     * P95 means: 95% of requests completed within this time.
     */
    public double getPercentile(double percentile) {
        if (latencies.isEmpty()) return 0.0;
        List<Double> sorted = new ArrayList<>(latencies);
        sorted.sort(Double::compareTo);
        int index = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
        index = Math.max(0, Math.min(index, sorted.size() - 1));
        return sorted.get(index);
    }

    /**
     * Returns min latency observed.
     */
    public double getMinLatency() {
        return latencies.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
    }

    /**
     * Returns max latency observed.
     */
    public double getMaxLatency() {
        return latencies.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
    }

    /**
     * Returns standard deviation of latencies.
     */
    public double getStdDev() {
        double mean = getMeanLatency();
        return Math.sqrt(latencies.stream()
                .mapToDouble(l -> (l - mean) * (l - mean))
                .average()
                .orElse(0.0));
    }

    /**
     * Returns cumulative regret — total extra latency paid vs. optimal.
     */
    public double getCumulativeRegret() { return cumulativeRegret; }

    /**
     * Returns total number of requests.
     */
    public int getTotalRequests() { return latencies.size(); }

    /**
     * Returns all recorded latencies (for charting).
     */
    public List<Double> getLatencies() { return new ArrayList<>(latencies); }

    /**
     * Returns the list of server selections for load distribution analysis.
     */
    public List<Integer> getServerSelections() { return new ArrayList<>(serverSelections); }

    public String getAlgorithmName() { return algorithmName; }

    /**
     * Prints a formatted summary report to stdout.
     */
    public void printSummary(int serverCount) {
        DoubleSummaryStatistics stats = latencies.stream()
                .mapToDouble(Double::doubleValue)
                .summaryStatistics();

        System.out.println("\n" + "=".repeat(60));
        System.out.printf("  ALGORITHM: %s%n", algorithmName);
        System.out.println("=".repeat(60));
        System.out.printf("  Total Requests  : %d%n", stats.getCount());
        System.out.printf("  Mean Latency    : %.2f ms%n", stats.getAverage());
        System.out.printf("  Min Latency     : %.2f ms%n", stats.getMin());
        System.out.printf("  Max Latency     : %.2f ms%n", stats.getMax());
        System.out.printf("  Std Dev         : %.2f ms%n", getStdDev());
        System.out.printf("  P50 (Median)    : %.2f ms%n", getPercentile(50));
        System.out.printf("  P95             : %.2f ms%n", getPercentile(95));
        System.out.printf("  P99             : %.2f ms%n", getPercentile(99));
        System.out.printf("  Cumulative Regret: %.2f ms%n", cumulativeRegret);

        // Server load distribution
        int[] counts = new int[serverCount];
        for (int s : serverSelections) {
            if (s >= 0 && s < serverCount) counts[s]++;
        }
        System.out.println("\n  Server Selection Distribution:");
        for (int i = 0; i < serverCount; i++) {
            double pct = 100.0 * counts[i] / latencies.size();
            System.out.printf("    Server-%d: %5d requests (%5.1f%%)%n", i, counts[i], pct);
        }
        System.out.println("=".repeat(60));
    }
}
