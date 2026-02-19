package com.loadbalancer.algorithm;

import com.loadbalancer.model.Server;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Random;

/**
 * ===========================================================================================
 * SOFTMAX LOAD BALANCER — Core Algorithm
 * ===========================================================================================
 *
 * THEORY:
 * -------
 * Softmax Action Selection is a technique from Reinforcement Learning (RL) that
 * converts a vector of Q-values (estimated rewards) into a probability distribution
 * using the softmax (Boltzmann) function:
 *
 *   P(server_i) = exp(Q_i / τ) / Σ_j exp(Q_j / τ)
 *
 * where τ (tau) is the "temperature" parameter:
 *   - τ → ∞  :  uniform distribution  (pure exploration)
 *   - τ → 0  :  argmax selection       (pure exploitation, i.e., greedy)
 *   - 0 < τ < ∞: interpolation between explore and exploit
 *
 * REWARD SIGNAL:
 * --------------
 * Since we want to MINIMIZE latency, we convert latency to reward:
 *   reward = -latency   (negative latency)
 *
 * Q-VALUES (Estimated Expected Reward):
 * ---------------------------------------
 * We maintain a running estimate of each server's average reward using
 * Exponential Moving Average (EMA) to handle the non-stationary distribution:
 *
 *   Q_i ← (1 - α) * Q_i + α * reward_i
 *
 * where α ∈ (0, 1] is the learning rate. Higher α gives more weight to recent
 * observations, making the algorithm adapt faster to performance changes.
 *
 * NUMERICAL STABILITY:
 * ---------------------
 * Naive computation of exp(Q_i / τ) can overflow for large Q_i values.
 * We use the log-sum-exp trick:
 *
 *   Let M = max(Q_i / τ)
 *   P(server_i) = exp(Q_i/τ - M) / Σ_j exp(Q_j/τ - M)
 *
 * This is mathematically equivalent but numerically stable since the largest
 * exponent is always exp(0) = 1.
 *
 * TEMPERATURE DECAY (Optional):
 * ------------------------------
 * We support linear temperature decay (cooling schedule):
 *   τ_t = max(τ_min, τ_0 - decay_rate * t)
 *
 * This starts with exploration and shifts toward exploitation over time.
 */
public class SoftmaxLoadBalancer implements LoadBalancer {

    private static final Logger log = LoggerFactory.getLogger(SoftmaxLoadBalancer.class);

    private final double initialTemperature;
    private final double minTemperature;
    private final double temperatureDecayRate;
    private final double learningRate;          // α — EMA learning rate
    private final Random random;

    private double[] qValues;                   // Q[i] = estimated reward for server i
    private int[]    selectionCounts;           // how many times each server was selected
    private double   currentTemperature;
    private int      stepCount;
    private int      serverCount;

    /**
     * @param serverCount         Number of servers in the cluster
     * @param initialTemperature  Initial τ — controls explore/exploit tradeoff
     * @param minTemperature      Minimum τ — prevents full greedy behavior
     * @param temperatureDecayRate Rate at which τ decreases per step (0 = no decay)
     * @param learningRate        α for EMA updates (0.0 < α ≤ 1.0)
     */
    public SoftmaxLoadBalancer(int serverCount,
                                double initialTemperature,
                                double minTemperature,
                                double temperatureDecayRate,
                                double learningRate) {
        this.serverCount = serverCount;
        this.initialTemperature = initialTemperature;
        this.minTemperature = minTemperature;
        this.temperatureDecayRate = temperatureDecayRate;
        this.learningRate = learningRate;
        this.random = new Random(12345L);

        initializeArrays();
    }

    private void initializeArrays() {
        this.qValues = new double[serverCount];
        this.selectionCounts = new int[serverCount];
        this.currentTemperature = initialTemperature;
        this.stepCount = 0;

        // Initialize Q-values with optimistic initial values (encourages early exploration)
        // We use 0.0 initially → all servers look equal → uniform exploration at start
        for (int i = 0; i < serverCount; i++) {
            qValues[i] = 0.0;
        }
    }

    @Override
    public int selectServer(List<Server> servers) {
        int n = servers.size();

        // Step 1: Compute scaled Q-values: Q_i / τ
        double[] scaledQ = new double[n];
        for (int i = 0; i < n; i++) {
            scaledQ[i] = qValues[i] / currentTemperature;
        }

        // Step 2: Numerical stability — subtract max (log-sum-exp trick)
        double maxVal = Double.NEGATIVE_INFINITY;
        for (double v : scaledQ) {
            if (v > maxVal) maxVal = v;
        }

        // Step 3: Compute exp(Q_i/τ - max) for each server
        double[] expValues = new double[n];
        double sumExp = 0.0;
        for (int i = 0; i < n; i++) {
            expValues[i] = Math.exp(scaledQ[i] - maxVal);
            sumExp += expValues[i];
        }

        // Step 4: Normalize to get probabilities
        double[] probabilities = new double[n];
        for (int i = 0; i < n; i++) {
            probabilities[i] = expValues[i] / sumExp;
        }

        // Step 5: Sample from the probability distribution
        int selectedIndex = sampleFromDistribution(probabilities);

        // Step 6: Update temperature (cooling schedule)
        selectionCounts[selectedIndex]++;
        stepCount++;
        currentTemperature = Math.max(minTemperature, initialTemperature - temperatureDecayRate * stepCount);

        if (log.isDebugEnabled()) {
            log.debug("[Softmax] Step={}, τ={:.4f}, selected=Server-{}, probs={}",
                    stepCount, currentTemperature, selectedIndex, formatProbs(probabilities));
        }

        return selectedIndex;
    }

    /**
     * Samples an index from a discrete probability distribution using inverse CDF.
     */
    private int sampleFromDistribution(double[] probabilities) {
        double sample = random.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (sample <= cumulative) {
                return i;
            }
        }
        // Fallback for floating point rounding errors
        return probabilities.length - 1;
    }

    @Override
    public void updateReward(int serverIndex, double latency) {
        // Convert latency to reward (minimize latency = maximize negative latency)
        // We normalize by dividing by 100 to keep Q-values in a reasonable range
        double reward = -latency / 100.0;

        // Exponential Moving Average update (handles non-stationary distributions)
        qValues[serverIndex] = (1.0 - learningRate) * qValues[serverIndex]
                              + learningRate * reward;
    }

    // --- Accessors for visualization / reporting ---

    public double[] getQValues() {
        return qValues.clone();
    }

    public double[] getProbabilities(int serverCount) {
        double[] scaledQ = new double[serverCount];
        for (int i = 0; i < serverCount; i++) {
            scaledQ[i] = qValues[i] / currentTemperature;
        }
        double maxVal = Double.NEGATIVE_INFINITY;
        for (double v : scaledQ) if (v > maxVal) maxVal = v;

        double[] expValues = new double[serverCount];
        double sumExp = 0.0;
        for (int i = 0; i < serverCount; i++) {
            expValues[i] = Math.exp(scaledQ[i] - maxVal);
            sumExp += expValues[i];
        }
        double[] probs = new double[serverCount];
        for (int i = 0; i < serverCount; i++) {
            probs[i] = expValues[i] / sumExp;
        }
        return probs;
    }

    public int[] getSelectionCounts() { return selectionCounts.clone(); }
    public double getCurrentTemperature() { return currentTemperature; }
    public int getStepCount() { return stepCount; }

    @Override
    public String getAlgorithmName() { return "Softmax (EMA, τ-decay)"; }

    @Override
    public void reset() {
        initializeArrays();
    }

    private String formatProbs(double[] probs) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < probs.length; i++) {
            sb.append(String.format("%.3f", probs[i]));
            if (i < probs.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}
