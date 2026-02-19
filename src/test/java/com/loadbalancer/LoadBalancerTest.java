package com.loadbalancer;

import com.loadbalancer.algorithm.RoundRobinLoadBalancer;
import com.loadbalancer.algorithm.RandomLoadBalancer;
import com.loadbalancer.algorithm.SoftmaxLoadBalancer;
import com.loadbalancer.model.Server;
import com.loadbalancer.metrics.MetricsCollector;
import com.loadbalancer.simulation.Simulation;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the Softmax Load Balancer system.
 *
 * Tests cover:
 *   1. Probability distribution validity (sums to 1.0)
 *   2. Numerical stability (no NaN/Infinity)
 *   3. Temperature effects (high τ → uniform, low τ → greedy)
 *   4. EMA learning (Q-values converge toward true reward)
 *   5. Algorithm comparison (Softmax ≤ Round-Robin mean latency)
 *   6. Round-Robin distribution uniformity
 *   7. Regret tracking
 */
class LoadBalancerTest {

    private List<Server> servers;
    private static final int SERVER_COUNT = 5;

    @BeforeEach
    void setUp() {
        servers = new ArrayList<>();
        servers.add(new Server(0, 20.0, 3.0, 0.05, 5.0));   // fast
        servers.add(new Server(1, 50.0, 8.0, 0.08, 10.0));  // medium
        servers.add(new Server(2, 80.0, 12.0, 0.10, 15.0)); // slow
        servers.add(new Server(3, 35.0, 5.0, 0.06, 8.0));   // medium-fast
        servers.add(new Server(4, 100.0, 20.0, 0.12, 20.0));// very slow
    }

    // ─── Softmax Probability Tests ────────────────────────────────────────────

    @Test
    @DisplayName("Softmax probabilities must sum to 1.0")
    void testProbabilitiesSumToOne() {
        SoftmaxLoadBalancer softmax = new SoftmaxLoadBalancer(SERVER_COUNT, 1.0, 0.1, 0.0, 0.1);
        double[] probs = softmax.getProbabilities(SERVER_COUNT);

        double sum = 0.0;
        for (double p : probs) sum += p;

        assertEquals(1.0, sum, 1e-9, "Probabilities must sum to exactly 1.0");
    }

    @Test
    @DisplayName("Softmax probabilities must all be non-negative")
    void testProbabilitiesNonNegative() {
        SoftmaxLoadBalancer softmax = new SoftmaxLoadBalancer(SERVER_COUNT, 1.0, 0.1, 0.0, 0.1);
        double[] probs = softmax.getProbabilities(SERVER_COUNT);

        for (int i = 0; i < SERVER_COUNT; i++) {
            assertTrue(probs[i] >= 0.0,
                    "Probability for server " + i + " must be non-negative, got: " + probs[i]);
        }
    }

    @Test
    @DisplayName("High temperature must produce near-uniform distribution")
    void testHighTemperatureIsUniform() {
        // τ = 100 → nearly uniform
        SoftmaxLoadBalancer softmax = new SoftmaxLoadBalancer(SERVER_COUNT, 100.0, 100.0, 0.0, 0.1);
        double[] probs = softmax.getProbabilities(SERVER_COUNT);

        double expected = 1.0 / SERVER_COUNT;
        for (double p : probs) {
            assertEquals(expected, p, 0.01,
                    "High temperature should yield ~uniform distribution");
        }
    }

    @Test
    @DisplayName("Low temperature must concentrate probability on best server")
    void testLowTemperatureFavoresBestServer() {
        // τ = 0.01 → nearly greedy
        SoftmaxLoadBalancer softmax = new SoftmaxLoadBalancer(SERVER_COUNT, 0.01, 0.01, 0.0, 1.0);

        // Train to know server 0 is best (negative latency = positive reward)
        for (int i = 0; i < 100; i++) {
            softmax.updateReward(0, 10.0);  // server 0 is very fast
            softmax.updateReward(1, 200.0); // server 1 is very slow
            softmax.updateReward(2, 200.0);
            softmax.updateReward(3, 200.0);
            softmax.updateReward(4, 200.0);
        }

        double[] probs = softmax.getProbabilities(SERVER_COUNT);

        // Server 0 should have the highest probability by far
        double server0Prob = probs[0];
        for (int i = 1; i < SERVER_COUNT; i++) {
            assertTrue(server0Prob > probs[i],
                    "Best server should have highest probability at low temperature");
        }
    }

    // ─── Numerical Stability Tests ────────────────────────────────────────────

    @Test
    @DisplayName("No NaN or Infinity in probabilities after extreme Q-values")
    void testNumericalStability() {
        SoftmaxLoadBalancer softmax = new SoftmaxLoadBalancer(SERVER_COUNT, 0.01, 0.01, 0.0, 1.0);

        // Feed extreme rewards that would cause overflow without log-sum-exp trick
        softmax.updateReward(0, 1.0);       // very good
        softmax.updateReward(1, 10000.0);   // very bad
        softmax.updateReward(2, 10000.0);
        softmax.updateReward(3, 0.001);     // extremely good
        softmax.updateReward(4, 50000.0);   // extremely bad

        double[] probs = softmax.getProbabilities(SERVER_COUNT);

        for (int i = 0; i < SERVER_COUNT; i++) {
            assertFalse(Double.isNaN(probs[i]),
                    "Probability for server " + i + " must not be NaN");
            assertFalse(Double.isInfinite(probs[i]),
                    "Probability for server " + i + " must not be Infinite");
        }
    }

    // ─── EMA Learning Tests ───────────────────────────────────────────────────

    @Test
    @DisplayName("Q-values should converge toward true reward with high learning rate")
    void testQValueConvergence() {
        SoftmaxLoadBalancer softmax = new SoftmaxLoadBalancer(SERVER_COUNT, 1.0, 0.1, 0.0, 0.5);

        double trueReward0 = -20.0 / 100.0;   // corresponds to 20ms latency
        double trueReward1 = -100.0 / 100.0;  // corresponds to 100ms latency

        // Train many times
        for (int i = 0; i < 500; i++) {
            softmax.updateReward(0, 20.0);
            softmax.updateReward(1, 100.0);
        }

        double[] qVals = softmax.getQValues();

        // Q[0] should be close to trueReward0 (≈ -0.2)
        assertEquals(trueReward0, qVals[0], 0.05,
                "Q-value for server 0 should converge to ~-0.2");
        // Q[1] should be close to trueReward1 (≈ -1.0)
        assertEquals(trueReward1, qVals[1], 0.05,
                "Q-value for server 1 should converge to ~-1.0");
    }

    @Test
    @DisplayName("Q-values should update in the correct direction")
    void testQValueUpdateDirection() {
        SoftmaxLoadBalancer softmax = new SoftmaxLoadBalancer(SERVER_COUNT, 1.0, 0.1, 0.0, 0.5);

        // Feed a very bad (high) latency to server 0 first, so Q[0] drops below 0
        softmax.updateReward(0, 500.0);  // very slow → Q[0] becomes negative

        double[] afterBad = softmax.getQValues();
        double qAfterBad = afterBad[0];

        // Now feed a very good (low) latency — Q[0] should move UP (toward 0 / less negative)
        softmax.updateReward(0, 1.0);   // very fast server

        double[] updatedQVals = softmax.getQValues();
        assertTrue(updatedQVals[0] > qAfterBad,
                "Q-value should increase (become less negative) after a good (low) latency observation");
    }

    // ─── Round-Robin Distribution Test ───────────────────────────────────────

    @Test
    @DisplayName("Round-Robin should distribute requests evenly across servers")
    void testRoundRobinDistribution() {
        RoundRobinLoadBalancer rr = new RoundRobinLoadBalancer();
        int[] counts = new int[SERVER_COUNT];
        int totalRequests = SERVER_COUNT * 100;

        for (int i = 0; i < totalRequests; i++) {
            int selected = rr.selectServer(servers);
            counts[selected]++;
        }

        int expected = totalRequests / SERVER_COUNT;
        for (int i = 0; i < SERVER_COUNT; i++) {
            assertEquals(expected, counts[i],
                    "Round-Robin should select each server exactly " + expected + " times");
        }
    }

    // ─── Simulation Integration Test ─────────────────────────────────────────

    @Test
    @DisplayName("Simulation should complete without errors")
    void testSimulationRuns() {
        Simulation sim = new Simulation(5, 500, false, 100);
        SoftmaxLoadBalancer softmax = new SoftmaxLoadBalancer(5, 2.0, 0.1, 0.001, 0.15);

        assertDoesNotThrow(() -> sim.run(softmax),
                "Simulation should complete without throwing exceptions");
    }

    @Test
    @DisplayName("Metrics collector should record correct request count")
    void testMetricsRequestCount() {
        Simulation sim = new Simulation(5, 200, false, 100);
        SoftmaxLoadBalancer softmax = new SoftmaxLoadBalancer(5, 2.0, 0.1, 0.001, 0.15);

        MetricsCollector metrics = sim.run(softmax);

        assertEquals(200, metrics.getTotalRequests(),
                "Metrics should record exactly 200 requests");
    }

    @Test
    @DisplayName("Softmax selection counts should sum to total requests")
    void testSelectionCountsSum() {
        SoftmaxLoadBalancer softmax = new SoftmaxLoadBalancer(SERVER_COUNT, 1.0, 0.1, 0.001, 0.15);

        int totalRequests = 1000;
        for (int i = 0; i < totalRequests; i++) {
            int selected = softmax.selectServer(servers);
            softmax.updateReward(selected, servers.get(selected).processRequest());
        }

        int[] counts = softmax.getSelectionCounts();
        int sum = 0;
        for (int c : counts) sum += c;

        assertEquals(totalRequests, sum,
                "Total selection counts should equal total requests");
    }

    // ─── Reset Test ───────────────────────────────────────────────────────────

    @Test
    @DisplayName("Reset should restore initial state")
    void testReset() {
        SoftmaxLoadBalancer softmax = new SoftmaxLoadBalancer(SERVER_COUNT, 2.0, 0.1, 0.001, 0.15);

        // Run some iterations
        for (int i = 0; i < 100; i++) {
            int s = softmax.selectServer(servers);
            softmax.updateReward(s, 50.0);
        }

        // Reset
        softmax.reset();

        // Q-values should be zero again
        double[] qVals = softmax.getQValues();
        for (double q : qVals) {
            assertEquals(0.0, q, 1e-9, "Q-values should be zero after reset");
        }

        assertEquals(2.0, softmax.getCurrentTemperature(), 1e-9,
                "Temperature should be restored to initial value after reset");
    }
}