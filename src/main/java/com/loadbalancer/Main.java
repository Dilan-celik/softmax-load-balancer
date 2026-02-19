package com.loadbalancer;

import com.loadbalancer.algorithm.*;
import com.loadbalancer.metrics.MetricsCollector;
import com.loadbalancer.simulation.Simulation;
import com.loadbalancer.ui.ConsoleVisualizer;

import java.util.ArrayList;
import java.util.List;

/**
 * ========================================================================================
 * SOFTMAX LOAD BALANCER — Main Entry Point
 * ========================================================================================
 *
 * Distributed Systems Assignment: Client-Side Load Balancer
 *
 * This program demonstrates and compares three load balancing strategies:
 *
 *   1. Round-Robin   — cycles through servers deterministically
 *   2. Random        — selects uniformly at random
 *   3. Softmax       — adaptive probabilistic selection using RL principles
 *
 * The simulation creates a cluster of K servers with NON-STATIONARY latency
 * distributions (latency changes over time) and injects degradation events
 * to test each algorithm's adaptability.
 *
 * Key Concepts Demonstrated:
 *   - Softmax (Boltzmann) action selection
 *   - Exponential Moving Average (EMA) for non-stationary reward estimation
 *   - Log-sum-exp trick for numerical stability
 *   - Temperature parameter (τ) for exploration vs. exploitation
 *   - Regret minimization
 *
 * Run Configuration (IntelliJ IDEA):
 *   1. Open project in IntelliJ IDEA
 *   2. Right-click Main.java → Run 'Main.main()'
 *   3. Or: mvn compile exec:java -Dexec.mainClass="com.loadbalancer.Main"
 * ========================================================================================
 */
public class Main {

    // ─── Simulation Parameters ───────────────────────────────────────────────
    private static final int SERVER_COUNT          = 5;
    private static final int TOTAL_REQUESTS        = 2000;
    private static final boolean ENABLE_EVENTS     = true;
    private static final int EVENT_INTERVAL        = 400;    // inject event every N requests

    // ─── Softmax Hyperparameters ──────────────────────────────────────────────
    private static final double INITIAL_TEMPERATURE  = 2.0;   // τ₀ — high = explore
    private static final double MIN_TEMPERATURE      = 0.1;   // τ_min — prevents full greedy
    private static final double TEMPERATURE_DECAY    = 0.001; // cool down rate per step
    private static final double LEARNING_RATE        = 0.15;  // α — EMA weight for recent obs

    public static void main(String[] args) {

        printBanner();

        System.out.println("  Configuration:");
        System.out.printf("    Servers             : %d%n", SERVER_COUNT);
        System.out.printf("    Total Requests      : %d%n", TOTAL_REQUESTS);
        System.out.printf("    Degradation Events  : %s (every %d requests)%n",
                ENABLE_EVENTS ? "ON" : "OFF", EVENT_INTERVAL);
        System.out.printf("    Softmax τ₀          : %.2f%n", INITIAL_TEMPERATURE);
        System.out.printf("    Softmax τ_min       : %.2f%n", MIN_TEMPERATURE);
        System.out.printf("    Softmax decay rate  : %.4f%n", TEMPERATURE_DECAY);
        System.out.printf("    EMA learning rate α : %.2f%n", LEARNING_RATE);

        // ─── Initialize Algorithms ────────────────────────────────────────────
        RoundRobinLoadBalancer roundRobin = new RoundRobinLoadBalancer();
        RandomLoadBalancer random = new RandomLoadBalancer();
        SoftmaxLoadBalancer softmax = new SoftmaxLoadBalancer(
                SERVER_COUNT,
                INITIAL_TEMPERATURE,
                MIN_TEMPERATURE,
                TEMPERATURE_DECAY,
                LEARNING_RATE
        );

        // ─── Create Simulation Engine ──────────────────────────────────────────
        Simulation simulation = new Simulation(
                SERVER_COUNT,
                TOTAL_REQUESTS,
                ENABLE_EVENTS,
                EVENT_INTERVAL
        );

        // ─── Run All Algorithms ────────────────────────────────────────────────
        List<MetricsCollector> allResults = new ArrayList<>();

        MetricsCollector rrMetrics     = simulation.run(roundRobin);
        MetricsCollector randomMetrics = simulation.run(random);
        MetricsCollector smMetrics     = simulation.run(softmax);

        allResults.add(rrMetrics);
        allResults.add(randomMetrics);
        allResults.add(smMetrics);

        // ─── Print Individual Reports ──────────────────────────────────────────
        System.out.println("\n\n" + "═".repeat(70));
        System.out.println("  DETAILED RESULTS");
        System.out.println("═".repeat(70));

        rrMetrics.printSummary(simulation.getServerCount());
        randomMetrics.printSummary(simulation.getServerCount());
        smMetrics.printSummary(simulation.getServerCount());

        // ─── Comparative Analysis ─────────────────────────────────────────────
        System.out.println("\n\n" + "═".repeat(70));
        System.out.println("  COMPARATIVE ANALYSIS");
        System.out.println("═".repeat(70));

        ConsoleVisualizer.printComparisonTable(allResults);
        ConsoleVisualizer.printLatencyBarChart(allResults);

        System.out.println("\n\n  Server Selection Distributions:");
        ConsoleVisualizer.printSelectionDistribution(rrMetrics, simulation.getServerCount());
        ConsoleVisualizer.printSelectionDistribution(randomMetrics, simulation.getServerCount());
        ConsoleVisualizer.printSelectionDistribution(smMetrics, simulation.getServerCount());

        // ─── Softmax Internal State ───────────────────────────────────────────
        System.out.println("\n\n  Softmax Algorithm — Final Internal State:");
        ConsoleVisualizer.printSoftmaxProbabilities(softmax, simulation.getServerCount());

        // ─── Latency Trend Sparklines ─────────────────────────────────────────
        System.out.println("\n");
        ConsoleVisualizer.printLatencyTrend(allResults, 50);

        // ─── Performance Improvement Summary ─────────────────────────────────
        printImprovementSummary(rrMetrics, randomMetrics, smMetrics);

        // ─── Algorithm Explanation ────────────────────────────────────────────
        printSoftmaxExplanation(softmax);

        System.out.println("\n\n  Simulation complete. See results above.");
    }

    private static void printImprovementSummary(
            MetricsCollector rr, MetricsCollector rand, MetricsCollector sm) {

        System.out.println("\n\n" + "═".repeat(70));
        System.out.println("  PERFORMANCE IMPROVEMENT (Softmax vs Baselines)");
        System.out.println("═".repeat(70));

        double smMean = sm.getMeanLatency();
        double rrMean = rr.getMeanLatency();
        double randMean = rand.getMeanLatency();

        double vsRR   = (rrMean   - smMean) / rrMean   * 100;
        double vsRand = (randMean - smMean) / randMean * 100;

        System.out.printf("  Softmax vs Round-Robin  : %+.1f%% mean latency improvement%n", vsRR);
        System.out.printf("  Softmax vs Random       : %+.1f%% mean latency improvement%n", vsRand);

        double rrRegret   = rr.getCumulativeRegret();
        double randRegret = rand.getCumulativeRegret();
        double smRegret   = sm.getCumulativeRegret();

        System.out.printf("%n  Cumulative Regret Reduction:%n");
        System.out.printf("    vs Round-Robin : %.1f ms less total wait time%n", rrRegret - smRegret);
        System.out.printf("    vs Random      : %.1f ms less total wait time%n", randRegret - smRegret);

        System.out.println("\n  Interpretation:");
        System.out.println("    Softmax learns to prefer low-latency servers over time.");
        System.out.println("    After degradation events, it adapts via EMA reward updates.");
        System.out.println("    Round-Robin and Random remain blind to server performance.");
    }

    private static void printSoftmaxExplanation(SoftmaxLoadBalancer softmax) {
        System.out.println("\n\n" + "═".repeat(70));
        System.out.println("  SOFTMAX FORMULA REFERENCE");
        System.out.println("═".repeat(70));
        System.out.println();
        System.out.println("  Selection Probability:");
        System.out.println("  ┌──────────────────────────────────────────────────┐");
        System.out.println("  │                                                  │");
        System.out.println("  │        exp(Q_i / τ)                              │");
        System.out.println("  │  P_i = ─────────────────                         │");
        System.out.println("  │        Σ_j exp(Q_j / τ)                          │");
        System.out.println("  │                                                  │");
        System.out.println("  │  where: Q_i = EMA reward for server i            │");
        System.out.println("  │         τ   = temperature (current: "
                + String.format("%.4f", softmax.getCurrentTemperature()) + ")        │");
        System.out.println("  │                                                  │");
        System.out.println("  └──────────────────────────────────────────────────┘");
        System.out.println();
        System.out.println("  Numerical Stability (Log-Sum-Exp Trick):");
        System.out.println("  ┌──────────────────────────────────────────────────┐");
        System.out.println("  │  Let M = max(Q_i / τ)                            │");
        System.out.println("  │                                                  │");
        System.out.println("  │        exp(Q_i/τ - M)          ← always ≤ 1     │");
        System.out.println("  │  P_i = ─────────────────────                     │");
        System.out.println("  │        Σ_j exp(Q_j/τ - M)                        │");
        System.out.println("  │                                                  │");
        System.out.println("  │  Prevents overflow: largest exp() = exp(0) = 1  │");
        System.out.println("  └──────────────────────────────────────────────────┘");
        System.out.println();
        System.out.println("  EMA Update Rule (Non-Stationary Adaptation):");
        System.out.println("  ┌──────────────────────────────────────────────────┐");
        System.out.println("  │  Q_i ← (1 - α) × Q_i + α × reward_i            │");
        System.out.println("  │                                                  │");
        System.out.println("  │  reward_i = -latency_i / 100                    │");
        System.out.println("  │  α = learning rate (higher α → adapts faster)   │");
        System.out.println("  └──────────────────────────────────────────────────┘");
    }

    private static void printBanner() {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                                                                      ║");
        System.out.println("║     ███████  ██████  ███████ ████████ ███    ███  █████  ██   ██    ║");
        System.out.println("║     ██      ██    ██ ██         ██    ████  ████ ██   ██  ██ ██     ║");
        System.out.println("║     ███████ ██    ██ █████      ██    ██ ████ ██ ███████   ███      ║");
        System.out.println("║          ██ ██    ██ ██         ██    ██  ██  ██ ██   ██  ██ ██     ║");
        System.out.println("║     ███████  ██████  ██         ██    ██      ██ ██   ██ ██   ██    ║");
        System.out.println("║                                                                      ║");
        System.out.println("║         LOAD BALANCER  |  Softmax Action Selection                  ║");
        System.out.println("║         Distributed Systems — Client-Side Load Balancer              ║");
        System.out.println("║                                                                      ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════╝");
        System.out.println();
    }
}
