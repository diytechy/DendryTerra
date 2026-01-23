Claude’s Plan
DendryTerra Implementation Plan
Overview
Create a Terra addon that implements Dendry noise generation, porting the C++ algorithm from C:\Projects\DendryNoise to Java following Terra's addon conventions.

Project Structure

c:\Projects\DendryTerra\
├── build.gradle.kts
├── settings.gradle.kts
├── gradlew.bat / gradlew
├── gradle/wrapper/
│   ├── gradle-wrapper.jar
│   └── gradle-wrapper.properties
├── src/main/
│   ├── java/com/dfsek/terra/addons/dendry/
│   │   ├── DendryAddon.java          # Entry point (AddonInitializer)
│   │   ├── DendrySampler.java        # Core algorithm (implements Sampler)
│   │   ├── DendryTemplate.java       # Config template (@Value annotations)
│   │   ├── DendryReturnType.java     # Enum: DISTANCE, WEIGHTED, ELEVATION
│   │   └── math/
│   │       ├── Point2D.java          # 2D point with operations
│   │       ├── Point3D.java          # 3D point (x,y = position, z = elevation)
│   │       ├── Vec2D.java            # 2D vector with dot, normalize, rotate
│   │       ├── Segment2D.java        # 2D line segment
│   │       ├── Segment3D.java        # 3D line segment with elevation
│   │       ├── MathUtils.java        # lerp, clamp, distance calculations
│   │       └── CatmullRomSpline.java # Spline subdivision for smoothing
│   └── resources/
│       └── terra.addon.yml           # Addon manifest
└── LICENSE
Files to Create
1. build.gradle.kts

plugins {
    java
}

group = "com.dfsek.terra.addons"
version = "1.0.0"

repositories {
    mavenCentral()
    maven { url = uri("https://maven.solo-studios.ca/releases") }
}

dependencies {
    compileOnly("com.dfsek.terra:manifest-addon-loader:1.0.0-BETA+a159debe3")
    compileOnly("com.dfsek:seismic:0.8.2")
    compileOnly("com.dfsek.terra:base:7.0.0-BETA+a159debe3")
    compileOnly("com.dfsek.tectonic:common:4.2.1")
    compileOnly("org.slf4j:slf4j-api:2.0.9")
}

java {
    toolchain { languageVersion.set(JavaLanguageVersion.of(21)) }
}

tasks.jar {
    archiveBaseName.set("DendryTerra")
}
2. settings.gradle.kts

rootProject.name = "DendryTerra"
3. terra.addon.yml

schema-version: 1
contributors:
  - DendryTerra Contributors
id: dendry-noise
version: 1.0.0
entrypoints:
  - "com.dfsek.terra.addons.dendry.DendryAddon"
license: GNU General Public License v3.0
4. DendryAddon.java
Implements AddonInitializer
Uses @Inject for Platform and BaseAddon
Registers DENDRY sampler in ConfigPackPreLoadEvent
Registers DendryReturnType loader
Pattern: NoiseAddon.java
5. DendryTemplate.java
Extends appropriate base template or implements ObjectTemplate<Sampler>
Configuration fields with @Value and @Default:
n (int, default 2): Resolution levels 1-5
epsilon (double, default 0): Point bias [0, 0.5)
delta (double, default 0.05): Displacement amount
slope (double, default 0.5): Valley/cliff factor
frequency (double, default 0.001): Grid scaling
return (DendryReturnType, default ELEVATION)
sampler (Sampler, optional): Control function
salt (long, default 0): Seed modifier
Pattern: PseudoErosionTemplate.java
6. DendrySampler.java (Core Algorithm)
Implements com.dfsek.seismic.type.sampler.Sampler with methods:

getSample(double x, double z) - 2D sampling
getSample(double x, double y, double z) - 3D sampling (use x,z plane)
Algorithm Flow (ported from noise.h):

Scale input coordinates by frequency
Determine grid cell at resolution level 1
Generate neighboring points (3x3 grid) with epsilon bias
Evaluate control function for elevation at each point
Create segments connecting each point to lowest neighbor
Subdivide segments using Catmull-Rom splines (4 subdivisions)
Displace segment midpoints by delta amount perpendicular to direction
Repeat for resolution levels 2 through n (cell size halves each level)
Compute result based on return type:
DISTANCE: Euclidean distance to nearest segment
WEIGHTED: Distance weighted by branch level
ELEVATION: Interpolated Z from nearest segment
Key Methods to Port:

GeneratePoint() - Deterministic point in cell using epsilon
GenerateSegments() - Connect to lowest neighbor
SubdivideSegments() - Catmull-Rom smoothing
DisplaceSegments() - Perpendicular perturbation
NearestSegmentProjectionZ() - Find closest segment
7. Math Classes
Port from math2d.h, math3d.h, spline.cpp:

Point2D/Point3D: Immutable with add, subtract, scale, distance
Vec2D: Direction vector with dot, normalize, rotateCCW90
Segment2D/Segment3D: Line segments with midpoint, length
MathUtils: lerp, clamp, pointLineProjection, distanceToSegment
CatmullRomSpline: Chordal parameterization for smooth subdivision
Implementation Order
Setup Gradle project - build.gradle.kts, settings.gradle.kts, wrapper
Create math utilities - Point2D, Point3D, Vec2D, Segment classes, MathUtils
Implement CatmullRomSpline - Subdivision algorithm
Create DendryReturnType enum
Implement DendrySampler - Core algorithm (start with resolution=1)
Create DendryTemplate - Configuration binding
Create DendryAddon - Registration and initialization
Create terra.addon.yml - Manifest
Build and test - Place JAR in NoiseTool addons folder
Verification
Run gradlew.bat build to compile
Copy build/libs/DendryTerra-1.0.0.jar to C:\Projects\NoiseTool\build\libs\addons\
Start NoiseTool and verify console shows addon loaded
Test with YAML config:

sampler:
  type: DENDRY
  n: 2
  epsilon: 0
  delta: 0.05
  frequency: 0.001
  return: elevation
Visualize output to confirm branching patterns
Key Reference Files
File	Purpose
NoiseAddon.java	Addon registration pattern
PseudoErosionTemplate.java	Template with sampler field
noise.h	Core C++ algorithm
spline.cpp	Catmull-Rom implementation
math2d.cpp	Distance calculations
