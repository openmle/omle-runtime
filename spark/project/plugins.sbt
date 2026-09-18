// sbt-ci-release bundles sbt-dynver (version from the git tag), sbt-pgp
// (signing) and sbt-sonatype (upload), which is every piece Maven Central
// requires. 1.11.0+ targets the Central Publisher Portal directly — Sonatype
// retired the legacy OSSRH endpoint in June 2025 — and needs sbt 1.11 or newer,
// which is why project/build.properties was raised from 1.9.9.
//
// On a tagged commit `sbt ci-release` publishes a signed release; anywhere else
// it publishes a snapshot. Version numbering is dynver's, so nothing here has
// to be edited at release time.
addSbtPlugin("com.github.sbt" % "sbt-ci-release" % "1.12.1")
