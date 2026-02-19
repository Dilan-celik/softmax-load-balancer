package com.loadbalancer.algorithm;

import com.loadbalancer.model.Server;

import java.util.List;

/**
 * Common interface for all load balancing algorithms.
 * Implementations include: SoftmaxLoadBalancer, RoundRobinLoadBalancer, RandomLoadBalancer.
 */
public interface LoadBalancer {

    /**
     * Selects a server from the cluster to handle the next request.
     *
     * @param servers List of available servers
     * @return Index of the selected server
     */
    int selectServer(List<Server> servers);

    /**
     * Updates the internal state of the algorithm with observed latency.
     * Only meaningful for learning algorithms (e.g., Softmax with UCB or EMA).
     *
     * @param serverIndex Index of the server that handled the request
     * @param latency     Observed latency in milliseconds
     */
    void updateReward(int serverIndex, double latency);

    /**
     * Returns the name of this algorithm for logging/reporting.
     */
    String getAlgorithmName();

    /**
     * Resets internal state (for fresh simulation runs).
     */
    void reset();
}
