plugins {
    java
}

group = "dendryterra"
version = "1.0.0"

repositories {
    mavenCentral()
    maven {
        name = "Solo Studios"
        url = uri("https://maven.solo-studios.ca/releases")
    }
}

dependencies {
    compileOnly("com.dfsek.terra:manifest-addon-loader:1.0.0-BETA+a159debe3")
    compileOnly("com.dfsek:seismic:0.8.2")
    compileOnly("com.dfsek.terra:base:7.0.0-BETA+a159debe3")
    compileOnly("com.dfsek.tectonic:common:4.2.1")
    compileOnly("org.slf4j:slf4j-api:2.0.9")
    compileOnly("com.github.ben-manes.caffeine:caffeine:3.1.8")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(23))
    }
}

tasks.jar {
    archiveBaseName.set("DendryTerra")
}
