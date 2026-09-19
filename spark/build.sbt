name := "omle-spark"

// Maven Central rejects a release without homepage, licenses, developers and
// scmInfo. sbt-ci-release reads them from ThisBuild, and supplies the version
// itself from the git tag — do not set `version` here.
inThisBuild(List(
  organization := "io.github.openmle",
  homepage     := Some(url("https://github.com/openmle/omle-runtime")),
  licenses     := List(
    "Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0.txt")),
  developers   := List(
    Developer("openmle", "OMLE", "", url("https://github.com/openmle"))),
  scmInfo      := Some(ScmInfo(
    url("https://github.com/openmle/omle-runtime"),
    "scm:git:https://github.com/openmle/omle-runtime.git")),
))
// Scala 2.12 and 2.13 are binary-incompatible, and a Spark cluster can only
// load a jar built for the Scala version its own jars were built with — a
// mismatch fails at run time with NoSuchMethodError deep inside the Scala
// runtime, not at load time. Spark 4 dropped Scala 2.12 and Spark 3.5 is the
// last line that still publishes it, so the Spark version follows from the
// Scala version rather than being chosen independently:
//
//   Scala 2.12 -> Spark 3.5.x  (what `pip install pyspark~=3.5` bundles)
//   Scala 2.13 -> Spark 4.x    (what `pip install pyspark>=4` bundles)
//
// `sbt +package` builds both; `sbt ++2.12.18 package` builds just the one.
val scala212 = "2.12.18"
val scala213 = "2.13.14"

scalaVersion       := scala213
crossScalaVersions := Seq(scala212, scala213)

def sparkVersionFor(scalaVer: String): String =
  CrossVersion.partialVersion(scalaVer) match {
    case Some((2, 12)) => "3.5.3"
    case _             => "4.0.2"
  }

libraryDependencies ++= {
  val sparkVersion = sparkVersionFor(scalaVersion.value)
  Seq(
    "org.apache.spark" %% "spark-sql"   % sparkVersion % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
    "net.java.dev.jna"  % "jna"         % "5.14.0",
    // Test scope re-declares Spark so tests run without a cluster
    "org.apache.spark" %% "spark-sql"   % sparkVersion % Test,
    "org.apache.spark" %% "spark-mllib" % sparkVersion % Test,
    "org.scalatest"    %% "scalatest"   % "3.2.17"     % Test,
  )
}

// The omle-runtime Java bindings jar — build first with `mvn package` in java/.
// Matched by glob rather than by name: both projects now take their version
// from the git tag, so the filename is not known ahead of time. The sources and
// javadoc jars that a release build also produces are excluded.
Compile / unmanagedJars ++= {
  val dir = (baseDirectory.value / ".." / "java" / "target").getAbsoluteFile
  val jars = Option(dir.listFiles()).getOrElse(Array.empty).filter { f =>
    f.getName.startsWith("omle-runtime-") && f.getName.endsWith(".jar") &&
      !f.getName.endsWith("-sources.jar") && !f.getName.endsWith("-javadoc.jar")
  }
  // Newest by timestamp, not by name: version strings do not sort lexically.
  jars.sortBy(_.lastModified).lastOption.map(Attributed.blank(_)).toSeq
}

Test / fork        := true
Test / javaOptions ++= Seq(
  s"-Djna.library.path=${(baseDirectory.value / ".." / "python" / "omle_runtime").getAbsolutePath}",
  // Spark 3.5 requires access to internal JDK classes on Java 17+
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
)
