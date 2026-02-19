package com.loadbalancer.model;

/**
 * Represents a client request dispatched to the cluster.
 */
public class Request {

    private final int requestId;
    private final long timestamp;
    private int assignedServerId;
    private double latency;
    private boolean completed;

    public Request(int requestId) {
        this.requestId = requestId;
        this.timestamp = System.currentTimeMillis();
        this.completed = false;
    }

    public void complete(int serverId, double latency) {
        this.assignedServerId = serverId;
        this.latency = latency;
        this.completed = true;
    }

    // --- Getters ---

    public int getRequestId() { return requestId; }
    public long getTimestamp() { return timestamp; }
    public int getAssignedServerId() { return assignedServerId; }
    public double getLatency() { return latency; }
    public boolean isCompleted() { return completed; }

    @Override
    public String toString() {
        return String.format("Request[id=%d, server=%d, latency=%.2fms]",
                requestId, assignedServerId, latency);
    }
}
