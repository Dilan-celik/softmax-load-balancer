package com.loadbalancer.simulation;

import com.loadbalancer.algorithm.LoadBalancer;
import com.loadbalancer.metrics.MetricsCollector;
import com.loadbalancer.model.Server;

import java.util.ArrayList;
import java.util.List;

/**
 * Simulation Engine
 *
 * Orchestrates a complete load balancing simulation:
 * 1. Creates a cluster of K servers with non-stationary latency distributions
 * 2. Runs N requests through the specified load balancer algorithm
 * 3. Injects degradation/recovery events at configurable intervals
 * 4. Collects and returns metrics for analysis
 */
public class Simulation {

    // --- Simulation Configuration ---
    private final int serverCount;
    private final int totalRequests;
    private final boolean enableDegradationEvents;
    private final int degradationInterval;     // every N requests, inject a degradation
    private final double optimalLatency;       // theoretical best (for regret calculation)

    // Cluster configuration (can be customized)
    private List<Server> servers;

    public Simulation(int serverCount,
                      int totalRequests,
                      boolean enableDegradationEvents,
                      int degradationInterval) {
        this.serverCount = serverCount;
        this.totalRequests = totalRequests;
        this.enableDegradationEvents = enableDegradationEvents;
        this.degradationInterval = degradationInterval;
        this.optimalLatency = 20.0; // approximate best-case latency in our setup
        this.servers = createDefaultCluster();
    }

    /**
     * Creates a diverse cluster where servers have different base latencies
     * and non-stationary drift patterns to simulate real-world heterogeneity.
     */
    private List<Server> createDefaultCluster() {
        List<Server> cluster = new ArrayList<>();

        //          id  baseLatency  noise  driftRate  driftAmplitude
        cluster.add(new Server(0,  20.0,   5.0,   0.05,   10.0));   // Fast, stable
        cluster.add(new Server(1,  50.0,  10.0,   0.08,   20.0));   // Medium, moderate drift
        cluster.add(new Server(2,  80.0,  15.0,   0.12,   30.0));   // Slow, high variance
        cluster.add(new Server(3,  35.0,   8.0,   0.07,   15.0));   // Medium-fast
        cluster.add(new Server(4, 100.0,  20.0,   0.15,   40.0));   // Slow, very noisy

        return cluster.subList(0, Math.min(serverCount, cluster.size()));
    }

    /**
     * Runs the simulation with the given load balancer algorithm.
     *
     * @param loadBalancer The algorithm to test
     * @return MetricsCollector containing all recorded results
     */
    public MetricsCollector run(LoadBalancer loadBalancer) {
        // Reset the algorithm and re-create fresh servers for fair comparison
        loadBalancer.reset();
        servers = createDefaultCluster();

        MetricsCollector metrics = new MetricsCollector(
                loadBalancer.getAlgorithmName(),
                optimalLatency,
                100  // rolling window size
        );

        System.out.printf("%n>>> Running simulation: %s (%d requests, %d servers)%n",
                loadBalancer.getAlgorithmName(), totalRequests, servers.size());

        long startTime = System.currentTimeMillis();

        for (int reqId = 0; reqId < totalRequests; reqId++) {

            // Inject degradation event periodically (simulates real-world incidents)
            if (enableDegradationEvents && reqId > 0 && reqId % degradationInterval == 0) {
                injectDegradationEvent(reqId);
            }

            // 1. Select a server
            int selectedIndex = loadBalancer.selectServer(servers);
            Server selectedServer = servers.get(selectedIndex);

            // 2. Process request (observe latency)
            double latency = selectedServer.processRequest();

            // 3. Feed the reward back into the algorithm
            loadBalancer.updateReward(selectedIndex, latency);

            // 4. Record metrics
            metrics.record(selectedIndex, latency);

            // Progress indicator for long simulations
            if ((reqId + 1) % (totalRequests / 10) == 0) {
                int progress = (int)(100.0 * (reqId + 1) / totalRequests);
                System.out.printf("    [%3d%%] Rolling avg latency: %.2f ms%n",
                        progress, metrics.getRollingAverage());
            }
        }

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.printf("    Simulation complete in %d ms%n", elapsed);

        return metrics;
    }

    /**
     * Injects a random degradation or recovery event into the cluster.
     * This tests how quickly each algorithm adapts.
     */
    private void injectDegradationEvent(int requestId) {
        // Randomly select a server to degrade or recover
        int targetIndex = (int)(Math.random() * servers.size());
        Server target = servers.get(targetIndex);

        // Alternate between degradation and recovery
        boolean isDegradation = (requestId / degradationInterval) % 2 == 0;

        if (isDegradation) {
            target.simulateDegradation(1.5);
            System.out.printf("  [Event @ req %d] DEGRADATION: Server-%d latency increased 50%%!%n",
                    requestId, targetIndex);
        } else {
            target.simulateRecovery(1.5);
            System.out.printf("  [Event @ req %d] RECOVERY:    Server-%d latency normalized%n",
                    requestId, targetIndex);
        }
    }

    /**
     * Returns the servers for external inspection.
     */
    public List<Server> getServers() { return servers; }

    public int getServerCount() { return servers.size(); }
}
