package com.loadbalancer.model;

import java.util.Random;

/**
 * Represents a server in the distributed cluster.
 * Each server has a non-stationary (time-varying) latency distribution
 * to simulate real-world performance fluctuations.
 */
public class Server {

    private final int id;
    private final String name;
    private final Random random;

    // Base latency in milliseconds (changes over time to simulate non-stationary behavior)
    private double baseLatency;
    private double noiseFactor;

    // Drift parameters to simulate gradual performance changes
    private double driftRate;
    private double driftAmplitude;
    private int tickCount;

    // Statistics tracking
    private int totalRequests;
    private double totalLatency;

    public Server(int id, double baseLatency, double noiseFactor, double driftRate, double driftAmplitude) {
        this.id = id;
        this.name = "Server-" + id;
        this.baseLatency = baseLatency;
        this.noiseFactor = noiseFactor;
        this.driftRate = driftRate;
        this.driftAmplitude = driftAmplitude;
        this.random = new Random(id * 42L); // reproducible randomness per server
        this.tickCount = 0;
        this.totalRequests = 0;
        this.totalLatency = 0.0;
    }

    /**
     * Simulates a request to this server.
     * Returns latency in milliseconds.
     *
     * Non-stationary behavior:
     *   latency = baseLatency + drift(t) + noise
     *
     * drift(t) = driftAmplitude * sin(driftRate * t)
     * noise    = Gaussian(0, noiseFactor)
     */
    public double processRequest() {
        tickCount++;

        // Compute time-varying drift (sinusoidal pattern simulates day/night cycles etc.)
        double drift = driftAmplitude * Math.sin(driftRate * tickCount);

        // Add Gaussian noise for realistic fluctuation
        double noise = random.nextGaussian() * noiseFactor;

        // Ensure latency is always positive
        double latency = Math.max(1.0, baseLatency + drift + noise);

        // Track statistics
        totalRequests++;
        totalLatency += latency;

        return latency;
    }

    /**
     * Simulates a "degradation event" (e.g., GC pause, hot restart).
     * Temporarily increases baseLatency.
     */
    public void simulateDegradation(double factor) {
        this.baseLatency *= factor;
    }

    /**
     * Simulates a "recovery event" (e.g., scale-out, cache warmup).
     */
    public void simulateRecovery(double factor) {
        this.baseLatency /= factor;
    }

    // --- Getters ---

    public int getId() { return id; }
    public String getName() { return name; }
    public double getBaseLatency() { return baseLatency; }
    public int getTotalRequests() { return totalRequests; }
    public double getTotalLatency() { return totalLatency; }

    public double getAverageLatency() {
        return totalRequests == 0 ? 0.0 : totalLatency / totalRequests;
    }

    public int getTickCount() { return tickCount; }

    /**
     * Peek at expected latency without recording a request (for visualization).
     */
    public double peekCurrentBaseLatency() {
        double drift = driftAmplitude * Math.sin(driftRate * (tickCount + 1));
        return Math.max(1.0, baseLatency + drift);
    }

    @Override
    public String toString() {
        return String.format("Server[id=%d, baseLatency=%.1fms, avgLatency=%.1fms, requests=%d]",
                id, baseLatency, getAverageLatency(), totalRequests);
    }
}
