package org.fog.ft;

/**
 * Deterministic random draws keyed by the coordinates of the draw.
 *
 * <p>Every stochastic decision in the experiment harness (initial wear, wear noise,
 * shocks, sudden failures, telemetry dropout, failure instants) is a pure function of
 * {@code (seed, scenario, node name, epoch, stream)}. Two properties follow:
 *
 * <ul>
 *   <li><b>Order independence.</b> CloudSim does not deliver same-time events in a
 *       stable order between runs, so a shared {@link java.util.Random} sequence made
 *       the old harness irreproducible. A hashed draw does not depend on how many draws
 *       came before it.</li>
 *   <li><b>Independence between scenarios and seeds.</b> The previous injector keyed
 *       its draws on {@code (seed, nodeId, epoch)} only, and every scenario ran with the
 *       same seed and the same entity ids. The "random" failure of node 57 at epoch 31
 *       therefore recurred in every scenario: 92% of nodes that failed in two scenarios
 *       failed at the same epoch, and 140 of 165 held-out failures repeated a training
 *       failure. A model that recognised the node and the time could memorise them.
 *       The scenario index is now part of every key.</li>
 * </ul>
 *
 * <p>Nodes are keyed by <i>name</i> ({@code fog-123}, {@code edge-7}) rather than by
 * CloudSim entity id, so the same physical node receives the same exogenous randomness
 * under every fault-tolerance policy (common random numbers). Policies are then
 * compared on identical failure luck; only their own actions differ.
 */
public final class FtRandom {

    private FtRandom() {
    }

    /** The splitmix64 finaliser: a bijective 64-bit mixer with full avalanche. */
    public static long mix(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Stable 64-bit hash of a string: FNV-1a, then mixed. */
    public static long hash(String s) {
        long h = 0xCBF29CE484222325L;
        for (int i = 0; i < s.length(); i++) {
            h ^= s.charAt(i);
            h *= 0x100000001B3L;
        }
        return mix(h);
    }

    /** Combines the coordinates of one draw into a 64-bit key. */
    public static long key(long seed, int scenario, long nodeHash, long epoch, long stream) {
        long h = mix(seed ^ 0x9E3779B97F4A7C15L);
        h = mix(h ^ (scenario * 0xD1B54A32D192ED03L + 0x632BE59BD9B4E019L));
        h = mix(h ^ nodeHash);
        h = mix(h ^ (epoch * 0x9E3779B97F4A7C15L + 0x2545F4914F6CDD1DL));
        h = mix(h ^ (stream * 0xBF58476D1CE4E5B9L + 0x94D049BB133111EBL));
        return h;
    }

    /** Uniform draw in [0, 1) from a key (top 53 bits). */
    public static double uniform(long key) {
        return (mix(key) >>> 11) * 0x1.0p-53;
    }

    /** Standard normal draw from a key, by Box-Muller on two derived uniforms. */
    public static double normal(long key) {
        double u1 = uniform(key ^ 0x5851F42D4C957F2DL);
        double u2 = uniform(key ^ 0x14057B7EF767814FL);
        if (u1 < 1e-300) {
            u1 = 1e-300;
        }
        return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    }

    /** Named streams, so that no two kinds of decision ever share a draw. */
    public static final long S_INITIAL_HEALTH = 1;
    public static final long S_WEAR_NOISE = 2;
    public static final long S_SHOCK = 3;
    public static final long S_SHOCK_SIZE = 4;
    public static final long S_SUDDEN = 5;
    public static final long S_FAIL_OFFSET = 6;
    public static final long S_AMBIENT = 7;
    public static final long S_THERMAL_NOISE = 8;
    public static final long S_DROPOUT = 100;     // + channel index
    public static final long S_CRITICALITY = 200;
}
