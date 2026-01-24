#Build:

#cd /c/Projects/DendryTerra && JAVA_HOME="C:/JAVA/jdk-23" ./gradlew build 2>&1

#Benchmark:

cd "c:/Projects/DendryTerra"
$Env:JAVA_HOME = 'C:/JAVA/jdk-23'
./gradlew benchmark --args="128" 2>&1