package org.fog.ft;

import java.util.Random;

import org.fog.application.selectivity.SelectivityModel;

/**
 * A reproducible drop-in for {@link org.fog.application.selectivity.FractionalSelectivity}.
 *
 * <p>iFogSim's own implementation decides whether to emit an output tuple with
 * {@code Math.random()}, which draws from a global generator nobody seeds. Any
 * application using a selectivity below 1.0 is therefore not reproducible: the same
 * simulation, same seed, same topology produces a different tuple stream every run.
 *
 * <p>That was not a theoretical concern here. Three runs of this harness at seed
 * 20260912 retained 6326, 6184 and 6355 telemetry snapshots and wrote three
 * different case files. Everything downstream inherited it -- queue depth varied on
 * a quarter of all rows, which moved uplink latency and jitter, which moved the
 * hazard, which changed which nodes failed. The simulator's own accounting was
 * bit-identical across those runs; this one call was the whole difference.
 *
 * <p>This class keeps the semantics exactly -- emit with probability p -- and draws
 * from a generator seeded from the run seed, so the tuple stream is identical run to
 * run. Implementing the interface rather than patching the original leaves every
 * existing iFogSim source file untouched.
 */
public class SeededFractionalSelectivity implements SelectivityModel {

    private final double selectivity;
    private final Random random;

    public SeededFractionalSelectivity(double selectivity, long seed) {
        this.selectivity = selectivity;
        this.random = new Random(seed);
    }

    @Override
    public boolean canSelect() {
        // A selectivity of 1.0 must always fire. nextDouble() returns [0,1), so the
        // comparison already guarantees that, but stating it makes the intent plain.
        if (selectivity >= 1.0) {
            return true;
        }
        return random.nextDouble() < selectivity;
    }

    @Override
    public double getMeanRate() {
        return selectivity;
    }

    @Override
    public double getMaxRate() {
        return selectivity;
    }

    public double getSelectivity() {
        return selectivity;
    }
}
