name := "watcher-trainer"
scalaVersion := "2.11.8"
resolvers += Resolver.sonatypeRepo("releases")

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "1.6.1" //% "provided"
  , "org.apache.spark" % "spark-mllib_2.11" % "1.6.1"// % "provided"
  , "com.github.melrief" % "purecsv_2.11" % "0.0.6"
  , "com.github.fommil.netlib" % "all" % "1.1.2"
)