plugins {
    java
    application
    `maven-publish`
}

group = "com.github.diytechy"
version = "1.0.0-BETA-E"

repositories {
    mavenCentral()
    maven {
        name = "Solo Studios"
        url = uri("https://maven.solo-studios.ca/releases")
    }
    maven {
        name = "Repsy-Terra"
        url = uri("https://repo.repsy.io/mvn/diytechy/terra")
    }
}

// For addon compilation (provided by Terra at runtime)
dependencies {
    compileOnly("com.dfsek.terra:manifest-addon-loader:1.0.0-BETA-ec788bf")
    compileOnly("com.dfsek:seismic:0.8.2")
    compileOnly("com.dfsek.terra:base:7.0.0-BETA-ec788bf")
    compileOnly("com.dfsek.terra:config-noise-function:1.2.0-BETA-ec788bf")
    compileOnly("com.dfsek.tectonic:common:4.2.1")
    compileOnly("org.slf4j:slf4j-api:2.0.9")
    compileOnly("com.github.ben-manes.caffeine:caffeine:3.1.8")
}

// Separate configuration for running benchmarks standalone
val benchmarkRuntimeOnly by configurations.creating {
    extendsFrom(configurations.compileOnly.get())
}

dependencies {
    benchmarkRuntimeOnly("com.dfsek:seismic:0.8.2")
    benchmarkRuntimeOnly("com.github.ben-manes.caffeine:caffeine:3.1.8")
    benchmarkRuntimeOnly("org.slf4j:slf4j-simple:2.0.9")
}

// Compile with JDK 25 for better compiler / JIT inlining decisions.
// --release 21 keeps the output bytecode at class version 65, so the addon jar
// remains loadable by Java 21+ Terra tools without recompilation.
java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(21) // Java 21 bytecode — compatible with current Terra runtime
}

application {
    mainClass.set("dendryterra.DendryBenchmarkRunner")
}

// Custom benchmark task that includes runtime dependencies.
// Override via Gradle project properties (use -P flag on the command line):
//   benchmarkGrid    — grid size (default 64)
//   benchmarkMode    — "far" (PIXEL_RIVER vs FAR comparison, default) or "all" (full suite)
//   benchmarkSpacing — world units between samples (default 1)
//   benchmarkAVX     — AVX level for JVM SIMD: 2 = AVX2 (default), 3 = AVX-512
//
// The benchmark process always runs on JDK 25 (via toolchain javaLauncher) even when
// the Gradle daemon uses an older JDK, so JIT and vectorization improvements apply.
tasks.register<JavaExec>("benchmark") {
    group = "verification"
    description = "Run DendrySampler performance benchmarks"
    classpath = sourceSets.main.get().output + configurations["benchmarkRuntimeOnly"]
    mainClass.set("dendryterra.DendryBenchmarkRunner")

    // Run the benchmark process on JDK 25 regardless of the Gradle daemon JDK
    javaLauncher.set(javaToolchains.launcherFor {
        languageVersion.set(JavaLanguageVersion.of(25))
    })

    val benchGrid    = project.findProperty("benchmarkGrid")    as String? ?: "64"
    val benchMode    = project.findProperty("benchmarkMode")    as String? ?: "far"
    val benchSpacing = project.findProperty("benchmarkSpacing") as String? ?: "1"
    val avxLevel     = project.findProperty("benchmarkAVX")     as String? ?: "2"
    args = listOf(benchGrid, benchMode, benchSpacing)

    // Enable SIMD vectorization. UseAVX=2 = AVX2 (256-bit, safe default).
    // UseAVX=3 = AVX-512 (512-bit) — only effective on CPUs that support it;
    // the JVM silently falls back if the ISA is unavailable.
    jvmArgs("-XX:UseAVX=$avxLevel")
}

tasks.jar {
    archiveBaseName.set("DendryTerra")
}

tasks.processResources {
    filesMatching("terra.addon.yml") {
        expand(mapOf("version" to project.version))
    }
}

publishing {
    repositories {
        mavenLocal()
        maven {
            name = "Repsy"
            url = uri("https://repo.repsy.io/mvn/diytechy/dendryterra")
            credentials {
                username = project.findProperty("repsy.user") as String? ?: System.getenv("REPSY_USERNAME")
                password = project.findProperty("repsy.key") as String? ?: System.getenv("REPSY_PASSWORD")
            }
        }
    }
    publications {
        create<MavenPublication>("repsy") {
            from(components["java"])
            artifactId = "dendryterra"
        }
    }
}

tasks.named("build") {
    finalizedBy(tasks.named("publishToMavenLocal"))
}

// Make 'run' task use benchmark dependencies too
tasks.named<JavaExec>("run") {
    classpath = sourceSets.main.get().output + configurations["benchmarkRuntimeOnly"]
}
