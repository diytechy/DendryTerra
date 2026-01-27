#Build:

#cd /c/Projects/DendryTerra && JAVA_HOME="C:/JAVA/jdk-23" ./gradlew build 2>&1

#Benchmark:

cd "c:/Projects/DendryTerra"
$Env:JAVA_HOME = 'C:/JAVA/jdk-23'
#C:/JAVA/jdk-23/bin/java.exe -cp DendryTerra.jar dendryterra.DendryBenchmarkRunner 32 2>&1

./gradlew benchmark --args="64" 2>&1