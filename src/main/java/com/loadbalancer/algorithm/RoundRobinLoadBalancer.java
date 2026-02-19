package com.loadbalancer.algorithm;

import com.loadbalancer.model.Server;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Round-Robin Load Balancer
 *
 * Classic algorithm that cycles through servers sequentially.
 * Distributes load evenly regardless of server performance.
 *
 * Pros:  Simple, predictable, O(1) selection
 * Cons:  Blind to server performance — will keep sending to slow servers
 */
public class RoundRobinLoadBalancer implements LoadBalancer {

    private final AtomicInteger counter;

    public RoundRobinLoadBalancer() {
        this.counter = new AtomicInteger(0);
    }

    @Override
    public int selectServer(List<Server> servers) {
        int selected = counter.getAndIncrement() % servers.size();
        if (counter.get() >= Integer.MAX_VALUE - 1) {
            counter.set(0);
        }
        return selected;
    }

    @Override
    public void updateReward(int serverIndex, double latency) {
        // Round Robin does not use feedback — intentionally blank
    }

    @Override
    public String getAlgorithmName() { return "Round-Robin"; }

    @Override
    public void reset() {
        counter.set(0);
    }
}
