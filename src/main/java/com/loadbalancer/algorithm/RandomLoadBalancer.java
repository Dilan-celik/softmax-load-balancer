package com.loadbalancer.algorithm;

import com.loadbalancer.model.Server;

import java.util.List;
import java.util.Random;

/**
 * Random Load Balancer
 *
 * Selects a server uniformly at random on each request.
 * Serves as a baseline for comparison.
 *
 * Pros:  Simple, no state needed, avoids correlated failures
 * Cons:  No adaptation to performance — same weakness as Round-Robin
 */
public class RandomLoadBalancer implements LoadBalancer {

    private final Random random;

    public RandomLoadBalancer() {
        this.random = new Random(99999L);
    }

    @Override
    public int selectServer(List<Server> servers) {
        return random.nextInt(servers.size());
    }

    @Override
    public void updateReward(int serverIndex, double latency) {
        // Random selection does not use feedback
    }

    @Override
    public String getAlgorithmName() { return "Random"; }

    @Override
    public void reset() {
        // Stateless — nothing to reset
    }
}
