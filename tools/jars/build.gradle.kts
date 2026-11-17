// Writes src/pyfinery/jvm/_runtime/jars.json: the jars pyfinery.jvm fetches from Maven Central on first use,
// per feature, each with its URL, size and SHA-256, and the platform of the jars of native libraries.
// Run it when Refinery is upgraded:  gradle -q -p tools/jars manifest  (needs a JDK 25)
import java.security.MessageDigest

// The java plugin registers how JVM versions and environments are matched when resolving.
plugins { java }

repositories { mavenCentral() }

val refinery = "0.3.0"
val slf4j = "2.0.17"

fun Configuration.forJar() {
    isCanBeConsumed = false
    // Refinery's build leaves log4j 1 out for its SLF4J bridge.
    exclude(group = "log4j", module = "log4j")
    exclude(group = "ch.qos.reload4j", module = "reload4j")
    attributes {
        attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
        attribute(Category.CATEGORY_ATTRIBUTE, objects.named(Category.LIBRARY))
        attribute(LibraryElements.LIBRARY_ELEMENTS_ATTRIBUTE, objects.named(LibraryElements.JAR))
        attribute(Bundling.BUNDLING_ATTRIBUTE, objects.named(Bundling.EXTERNAL))
        attribute(TargetJvmEnvironment.TARGET_JVM_ENVIRONMENT_ATTRIBUTE, objects.named(TargetJvmEnvironment.STANDARD_JVM))
        attribute(TargetJvmVersion.TARGET_JVM_VERSION_ATTRIBUTE, 25)
    }
}

// Parsing, compiling and matching rules. OR-tools comes in through scope reasoning, which matching never runs.
val validation by configurations.creating {
    forJar()
    exclude(group = "com.google.ortools")
}
// Generation: everything, with OR-tools, which solves scopes.
val generation by configurations.creating { forJar() }

dependencies {
    for (feature in listOf("validation", "generation")) {
        // The generator jar also holds ProblemLoader and the standalone injector, which matching needs.
        feature("tools.refinery:refinery-generator:$refinery")
        feature("tools.refinery:refinery-store-query-interpreter:$refinery")
        feature("org.slf4j:log4j-over-slf4j:$slf4j")
        feature("org.slf4j:slf4j-nop:$slf4j")
    }
}

fun sha256(file: File): String =
    MessageDigest.getInstance("SHA-256").digest(file.readBytes()).joinToString("") { "%02x".format(it) }

fun quoted(text: String?) = if (text == null) "null" else "\"$text\""

tasks.register("manifest") {
    val features = listOf(validation, generation)
    val target = rootDir.resolve("../../src/pyfinery/jvm/_runtime/jars.json")
    doLast {
        val out = StringBuilder("{\n  \"refinery\": \"$refinery\",\n  \"java\": 25,\n  \"features\": {\n")
        features.forEachIndexed { i, conf ->
            out.append("    \"${conf.name}\": [\n")
            val artifacts = conf.resolvedConfiguration.resolvedArtifacts.sortedBy { it.file.name }
            artifacts.forEachIndexed { j, a ->
                val id = a.moduleVersion.id
                val classifier = a.classifier?.let { "-$it" } ?: ""
                val path = "${id.group.replace('.', '/')}/${id.name}/${id.version}/" +
                    "${id.name}-${id.version}$classifier.${a.extension}"
                // OR-tools ships its native libraries in one jar per platform: a client fetches only its own.
                val platform = Regex("^ortools-(darwin|linux|win32)-(.+)$").find(id.name)
                    ?.let { "${it.groupValues[1]}-${it.groupValues[2]}" }
                out.append("      {\"file\": \"${a.file.name}\", ")
                out.append("\"url\": \"https://repo1.maven.org/maven2/$path\", ")
                out.append("\"size\": ${a.file.length()}, \"sha256\": \"${sha256(a.file)}\", ")
                out.append("\"platform\": ${quoted(platform)}}")
                out.append(if (j < artifacts.size - 1) ",\n" else "\n")
            }
            out.append(if (i < features.size - 1) "    ],\n" else "    ]\n")
        }
        out.append("  }\n}\n")
        target.writeText(out.toString())
        features.forEach { conf ->
            val files = conf.resolvedConfiguration.resolvedArtifacts.map { it.file }
            println("${conf.name}: ${files.size} jars, ${files.sumOf { it.length() } / 1_000_000} MB (all platforms)")
        }
    }
}
