package com.dfsek.terra.addons.dendry;

import com.dfsek.seismic.type.sampler.Sampler;
import com.dfsek.tectonic.api.config.template.ValidatedConfigTemplate;
import com.dfsek.tectonic.api.config.template.annotations.Default;
import com.dfsek.tectonic.api.config.template.annotations.Value;
import com.dfsek.tectonic.api.config.template.object.ObjectTemplate;
import com.dfsek.tectonic.api.exception.ValidationException;

import com.dfsek.terra.api.config.meta.Meta;

/**
 * Configuration template for the Dendry noise sampler.
 */
public class DendryTemplate implements ValidatedConfigTemplate, ObjectTemplate<Sampler> {

    @Value("n")
    @Default
    private @Meta int n = 2;

    @Value("epsilon")
    @Default
    private @Meta double epsilon = 0.0;

    @Value("delta")
    @Default
    private @Meta double delta = 0.05;

    @Value("slope")
    @Default
    private @Meta double slope = 0.5;

    @Value("frequency")
    @Default
    private @Meta double frequency = 0.001;

    @Value("return")
    @Default
    private @Meta DendryReturnType returnType = DendryReturnType.ELEVATION;

    @Value("sampler")
    @Default
    private @Meta Sampler controlSampler = null;

    @Value("salt")
    @Default
    private @Meta long salt = 0;

    @Override
    public boolean validate() throws ValidationException {
        if (n < 1 || n > 5) {
            throw new ValidationException("n must be between 1 and 5, got: " + n);
        }
        if (epsilon < 0 || epsilon >= 0.5) {
            throw new ValidationException("epsilon must be in range [0, 0.5), got: " + epsilon);
        }
        if (delta < 0) {
            throw new ValidationException("delta must be non-negative, got: " + delta);
        }
        return true;
    }

    @Override
    public Sampler get() {
        return new DendrySampler(
            n, epsilon, delta, slope, frequency,
            returnType, controlSampler, salt
        );
    }
}
