name         := "omle-spark"
organization := "io.github.openmle"
version      := "0.1.0"
scalaVersion := "2.13.14"

val sparkVersion = "4.0.2"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql"   % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
  "net.java.dev.jna"  % "jna"         % "5.14.0",
  // Test scope re-declares Spark so tests run without a cluster
  "org.apache.spark" %% "spark-sql"   % sparkVersion % Test,
  "org.apache.spark" %% "spark-mllib" % sparkVersion % Test,
  "org.scalatest"    %% "scalatest"   % "3.2.17"     % Test,
)

// The omle-runtime Java bindings jar — build first with `mvn package` in java/
Compile / unmanagedJars += Attributed.blank(
  (baseDirectory.value / ".." / "java" / "target" / "omle-runtime-0.1.0.jar").getAbsoluteFile
)

Test / fork        := true
Test / javaOptions ++= Seq(
  s"-Djna.library.path=${(baseDirectory.value / ".." / "python" / "omleruntime").getAbsolutePath}",
  // Spark 3.5 requires access to internal JDK classes on Java 17+
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
)
