plugins {
    java
    application
    `maven-publish`
}

group = "com.github.diytechy"
version = "1.0.0-BETA-1"

repositories {
    mavenCentral()
    maven {
        name = "Solo Studios"
        url = uri("https://maven.solo-studios.ca/releases")
    }
}

// For addon compilation (provided by Terra at runtime)
dependencies {
    compileOnly("com.dfsek.terra:manifest-addon-loader:1.0.0-BETA+a159debe3")
    compileOnly("com.dfsek:seismic:0.8.2")
    compileOnly("com.dfsek.terra:base:7.0.0-BETA+a159debe3")
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

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(23))
    }
}

application {
    mainClass.set("dendryterra.DendryBenchmarkRunner")
}

// Custom benchmark task that includes runtime dependencies
tasks.register<JavaExec>("benchmark") {
    group = "verification"
    description = "Run DendrySampler performance benchmarks"
    classpath = sourceSets.main.get().output + configurations["benchmarkRuntimeOnly"]
    mainClass.set("dendryterra.DendryBenchmarkRunner")

    // Pass command line args: ./gradlew benchmark --args="128"
    // Default grid size
    args = listOf("64")
}

tasks.jar {
    archiveBaseName.set("DendryTerra")
}

publishing {
    repositories {
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

// Make 'run' task use benchmark dependencies too
tasks.named<JavaExec>("run") {
    classpath = sourceSets.main.get().output + configurations["benchmarkRuntimeOnly"]
}
