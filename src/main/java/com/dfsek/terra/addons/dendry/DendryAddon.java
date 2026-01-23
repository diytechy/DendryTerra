package com.dfsek.terra.addons.dendry;

import com.dfsek.seismic.type.sampler.Sampler;
import com.dfsek.tectonic.api.config.template.object.ObjectTemplate;

import java.util.function.Supplier;

import com.dfsek.terra.addons.manifest.api.AddonInitializer;
import com.dfsek.terra.api.Platform;
import com.dfsek.terra.api.addon.BaseAddon;
import com.dfsek.terra.api.event.events.config.pack.ConfigPackPreLoadEvent;
import com.dfsek.terra.api.event.functional.FunctionalEventHandler;
import com.dfsek.terra.api.inject.annotations.Inject;
import com.dfsek.terra.api.registry.CheckedRegistry;
import com.dfsek.terra.api.util.reflection.TypeKey;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * DendryTerra addon entry point.
 * Registers the DENDRY sampler type for Terra configuration packs.
 */
public class DendryAddon implements AddonInitializer {
    private static final Logger LOGGER = LoggerFactory.getLogger(DendryAddon.class);

    public static final TypeKey<Supplier<ObjectTemplate<Sampler>>> NOISE_SAMPLER_TOKEN = new TypeKey<>() {};

    @Inject
    private Platform platform;

    @Inject
    private BaseAddon addon;

    @Override
    public void initialize() {
        LOGGER.info("Initializing DendryTerra addon...");

        platform.getEventManager()
            .getHandler(FunctionalEventHandler.class)
            .register(addon, ConfigPackPreLoadEvent.class)
            .then(event -> {
                CheckedRegistry<Supplier<ObjectTemplate<Sampler>>> noiseRegistry =
                    event.getPack().getOrCreateRegistry(NOISE_SAMPLER_TOKEN);

                // Register the DendryReturnType loader
                event.getPack().applyLoader(DendryReturnType.class,
                    (type, o, loader, depthTracker) -> DendryReturnType.valueOf(((String) o).toUpperCase()));

                // Register the DENDRY sampler type
                noiseRegistry.register(addon.key("DENDRY"), DendryTemplate::new);

                LOGGER.info("DendryTerra: Registered DENDRY sampler type");
            })
            .priority(50)
            .failThrough();

        LOGGER.info("DendryTerra addon initialized successfully");
    }
}
