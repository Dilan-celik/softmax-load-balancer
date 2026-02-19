package com.loadbalancer.ui;

import com.loadbalancer.metrics.MetricsCollector;
import com.loadbalancer.algorithm.SoftmaxLoadBalancer;

import java.util.List;

/**
 * Console-based visualization for simulation results.
 *
 * Renders ASCII bar charts and comparison tables to stdout,
 * making results interpretable directly in the terminal / IntelliJ Run console.
 */
public class ConsoleVisualizer {

    private static final int CHART_WIDTH = 50;
    private static final String BAR_CHAR = "█";
    private static final String HALF_BAR = "▌";

    /**
     * Prints a comparative summary table for all algorithms.
     */
    public static void printComparisonTable(List<MetricsCollector> results) {
        System.out.println("\n");
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                 LOAD BALANCER ALGORITHM COMPARISON                          ║");
        System.out.println("╠══════════════════╦═════════╦═════════╦═════════╦═════════╦═══════════════╣");
        System.out.println("║ Algorithm        ║ Mean    ║ P50     ║ P95     ║ P99     ║ Cum.Regret    ║");
        System.out.println("╠══════════════════╬═════════╬═════════╬═════════╬═════════╬═══════════════╣");

        for (MetricsCollector m : results) {
            System.out.printf("║ %-16s ║ %5.1fms ║ %5.1fms ║ %5.1fms ║ %5.1fms ║ %11.1fms ║%n",
                    truncate(m.getAlgorithmName(), 16),
                    m.getMeanLatency(),
                    m.getPercentile(50),
                    m.getPercentile(95),
                    m.getPercentile(99),
                    m.getCumulativeRegret());
        }

        System.out.println("╚══════════════════╩═════════╩═════════╩═════════╩═════════╩═══════════════╝");

        // Highlight winner
        MetricsCollector winner = results.stream()
                .min((a, b) -> Double.compare(a.getMeanLatency(), b.getMeanLatency()))
                .orElse(null);
        if (winner != null) {
            System.out.printf("%n  🏆  Best Mean Latency: %s (%.2f ms)%n",
                    winner.getAlgorithmName(), winner.getMeanLatency());
        }

        MetricsCollector lowestRegret = results.stream()
                .min((a, b) -> Double.compare(a.getCumulativeRegret(), b.getCumulativeRegret()))
                .orElse(null);
        if (lowestRegret != null) {
            System.out.printf("  🎯  Lowest Cumulative Regret: %s (%.2f ms)%n",
                    lowestRegret.getAlgorithmName(), lowestRegret.getCumulativeRegret());
        }
    }

    /**
     * Prints an ASCII bar chart comparing mean latencies across algorithms.
     */
    public static void printLatencyBarChart(List<MetricsCollector> results) {
        System.out.println("\n  Mean Latency Comparison (lower is better):");
        System.out.println("  " + "─".repeat(CHART_WIDTH + 30));

        double maxLatency = results.stream()
                .mapToDouble(MetricsCollector::getMeanLatency)
                .max()
                .orElse(1.0);

        for (MetricsCollector m : results) {
            double mean = m.getMeanLatency();
            int barLength = (int)(CHART_WIDTH * mean / maxLatency);
            String bar = BAR_CHAR.repeat(barLength);
            System.out.printf("  %-20s │%s %.1fms%n",
                    truncate(m.getAlgorithmName(), 20), bar, mean);
        }
        System.out.println("  " + "─".repeat(CHART_WIDTH + 30));
    }

    /**
     * Prints server selection distribution as a bar chart.
     */
    public static void printSelectionDistribution(MetricsCollector metrics, int serverCount) {
        System.out.printf("%n  Server Selection Distribution for [%s]:%n", metrics.getAlgorithmName());
        System.out.println("  " + "─".repeat(CHART_WIDTH + 25));

        List<Integer> selections = metrics.getServerSelections();
        int[] counts = new int[serverCount];
        for (int s : selections) {
            if (s >= 0 && s < serverCount) counts[s]++;
        }

        int total = selections.size();
        int maxCount = 0;
        for (int c : counts) if (c > maxCount) maxCount = c;

        for (int i = 0; i < serverCount; i++) {
            double pct = 100.0 * counts[i] / total;
            int barLength = maxCount == 0 ? 0 : (int)(CHART_WIDTH * counts[i] / maxCount);
            String bar = BAR_CHAR.repeat(barLength);
            System.out.printf("  Server-%-2d │ %s %5.1f%% (%d)%n", i, bar, pct, counts[i]);
        }
        System.out.println("  " + "─".repeat(CHART_WIDTH + 25));
    }

    /**
     * Prints the Softmax probability distribution for each server.
     */
    public static void printSoftmaxProbabilities(SoftmaxLoadBalancer softmax, int serverCount) {
        double[] probs = softmax.getProbabilities(serverCount);
        double[] qVals = softmax.getQValues();

        System.out.println("\n  Current Softmax State:");
        System.out.printf("  Temperature (τ): %.4f%n", softmax.getCurrentTemperature());
        System.out.println("  " + "─".repeat(60));
        System.out.printf("  %-10s %-15s %-15s %-20s%n",
                "Server", "Q-Value", "Probability", "Visual");
        System.out.println("  " + "─".repeat(60));

        for (int i = 0; i < serverCount; i++) {
            int barLen = (int)(40 * probs[i]);
            String bar = BAR_CHAR.repeat(barLen);
            System.out.printf("  Server-%-3d  Q=%8.4f     P=%6.3f    │%s%n",
                    i, qVals[i], probs[i], bar);
        }
        System.out.println("  " + "─".repeat(60));
    }

    /**
     * Prints a simplified rolling-window latency trend (ASCII sparkline).
     */
    public static void printLatencyTrend(List<MetricsCollector> results, int buckets) {
        System.out.println("\n  Latency Trend (time →, normalized to each algorithm's max):");

        for (MetricsCollector m : results) {
            List<Double> latencies = m.getLatencies();
            int total = latencies.size();
            if (total == 0) continue;

            int bucketSize = Math.max(1, total / buckets);
            double[] bucketAvgs = new double[buckets];
            double maxBucket = 0;

            for (int b = 0; b < buckets; b++) {
                int start = b * bucketSize;
                int end = Math.min(start + bucketSize, total);
                double sum = 0;
                for (int i = start; i < end; i++) sum += latencies.get(i);
                bucketAvgs[b] = (end > start) ? sum / (end - start) : 0;
                if (bucketAvgs[b] > maxBucket) maxBucket = bucketAvgs[b];
            }

            String[] sparkChars = {" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
            System.out.printf("  %-20s │", truncate(m.getAlgorithmName(), 20));
            for (int b = 0; b < buckets; b++) {
                int level = maxBucket == 0 ? 0 :
                        (int)(8 * bucketAvgs[b] / maxBucket);
                level = Math.max(0, Math.min(8, level));
                System.out.print(sparkChars[level]);
            }
            System.out.printf("│ avg=%.1fms%n", m.getMeanLatency());
        }
    }

    private static String truncate(String s, int maxLen) {
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen - 1) + "…";
    }
}
