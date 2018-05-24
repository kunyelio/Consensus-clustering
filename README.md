# Consensus-clustering
Consensus clustering example for Apache Spark v. 2.3

Download the following libraries from spark-2.3.0-hadoop2.7 binary release
(spark-2.3.0-bin-hadoop2.7/jars folder) and copy them under lib folder. Then
create an Eclipse Java project and import this directory. 

After 

JavaEWAH-0.3.2.jar
RoaringBitmap-0.5.11.jar
ST4-4.0.4.jar
activation-1.1.1.jar
aircompressor-0.8.jar
antlr-2.7.7.jar
antlr-runtime-3.4.jar
antlr4-runtime-4.7.jar
aopalliance-1.0.jar
aopalliance-repackaged-2.4.0-b34.jar
apache-log4j-extras-1.2.17.jar
apacheds-i18n-2.0.0-M15.jar
apacheds-kerberos-codec-2.0.0-M15.jar
api-asn1-api-1.0.0-M20.jar
api-util-1.0.0-M20.jar
arpack_combined_all-0.1.jar
arrow-format-0.8.0.jar
arrow-memory-0.8.0.jar
arrow-vector-0.8.0.jar
automaton-1.11-8.jar
avro-1.7.7.jar
avro-ipc-1.7.7.jar
avro-mapred-1.7.7-hadoop2.jar
base64-2.3.8.jar
bcprov-jdk15on-1.58.jar
bonecp-0.8.0.RELEASE.jar
breeze-macros_2.11-0.13.2.jar
breeze_2.11-0.13.2.jar
calcite-avatica-1.2.0-incubating.jar
calcite-core-1.2.0-incubating.jar
calcite-linq4j-1.2.0-incubating.jar
chill-java-0.8.4.jar
chill_2.11-0.8.4.jar
commons-beanutils-1.7.0.jar
commons-beanutils-core-1.8.0.jar
commons-cli-1.2.jar
commons-codec-1.10.jar
commons-collections-3.2.2.jar
commons-compiler-3.0.8.jar
commons-compress-1.4.1.jar
commons-configuration-1.6.jar
commons-crypto-1.0.0.jar
commons-dbcp-1.4.jar
commons-digester-1.8.jar
commons-httpclient-3.1.jar
commons-io-2.4.jar
commons-lang-2.6.jar
commons-lang3-3.5.jar
commons-logging-1.1.3.jar
commons-math3-3.4.1.jar
commons-net-2.2.jar
commons-pool-1.5.4.jar
compress-lzf-1.0.3.jar
core-1.1.2.jar
curator-client-2.7.1.jar
curator-framework-2.7.1.jar
curator-recipes-2.7.1.jar
datanucleus-api-jdo-3.2.6.jar
datanucleus-core-3.2.10.jar
datanucleus-rdbms-3.2.9.jar
derby-10.12.1.1.jar
eigenbase-properties-1.1.5.jar
flatbuffers-1.2.0-3f79e055.jar
generex-1.0.1.jar
gson-2.2.4.jar
guava-14.0.1.jar
guice-3.0.jar
guice-servlet-3.0.jar
hadoop-annotations-2.7.3.jar
hadoop-auth-2.7.3.jar
hadoop-client-2.7.3.jar
hadoop-common-2.7.3.jar
hadoop-hdfs-2.7.3.jar
hadoop-mapreduce-client-app-2.7.3.jar
hadoop-mapreduce-client-common-2.7.3.jar
hadoop-mapreduce-client-core-2.7.3.jar
hadoop-mapreduce-client-jobclient-2.7.3.jar
hadoop-mapreduce-client-shuffle-2.7.3.jar
hadoop-yarn-api-2.7.3.jar
hadoop-yarn-client-2.7.3.jar
hadoop-yarn-common-2.7.3.jar
hadoop-yarn-server-common-2.7.3.jar
hadoop-yarn-server-web-proxy-2.7.3.jar
hive-beeline-1.2.1.spark2.jar
hive-cli-1.2.1.spark2.jar
hive-exec-1.2.1.spark2.jar
hive-jdbc-1.2.1.spark2.jar
hive-metastore-1.2.1.spark2.jar
hk2-api-2.4.0-b34.jar
hk2-locator-2.4.0-b34.jar
hk2-utils-2.4.0-b34.jar
hppc-0.7.2.jar
htrace-core-3.1.0-incubating.jar
httpclient-4.5.4.jar
httpcore-4.4.8.jar
ivy-2.4.0.jar
jackson-annotations-2.6.7.jar
jackson-core-2.6.7.jar
jackson-core-asl-1.9.13.jar
jackson-databind-2.6.7.1.jar
jackson-dataformat-yaml-2.6.7.jar
jackson-jaxrs-1.9.13.jar
jackson-mapper-asl-1.9.13.jar
jackson-module-jaxb-annotations-2.6.7.jar
jackson-module-paranamer-2.7.9.jar
jackson-module-scala_2.11-2.6.7.1.jar
jackson-xc-1.9.13.jar
janino-3.0.8.jar
java-xmlbuilder-1.1.jar
javassist-3.18.1-GA.jar
javax.annotation-api-1.2.jar
javax.inject-1.jar
javax.inject-2.4.0-b34.jar
javax.servlet-api-3.1.0.jar
javax.ws.rs-api-2.0.1.jar
javolution-5.5.1.jar
jaxb-api-2.2.2.jar
jcl-over-slf4j-1.7.16.jar
jdo-api-3.0.1.jar
jersey-client-2.22.2.jar
jersey-common-2.22.2.jar
jersey-container-servlet-2.22.2.jar
jersey-container-servlet-core-2.22.2.jar
jersey-guava-2.22.2.jar
jersey-media-jaxb-2.22.2.jar
jersey-server-2.22.2.jar
jets3t-0.9.4.jar
jetty-6.1.26.jar
jetty-util-6.1.26.jar
jline-2.12.1.jar
joda-time-2.9.3.jar
jodd-core-3.5.2.jar
jpam-1.1.jar
json4s-ast_2.11-3.2.11.jar
json4s-core_2.11-3.2.11.jar
json4s-jackson_2.11-3.2.11.jar
jsp-api-2.1.jar
jsr305-1.3.9.jar
jta-1.1.jar
jtransforms-2.4.0.jar
jul-to-slf4j-1.7.16.jar
kryo-shaded-3.0.3.jar
kubernetes-client-3.0.0.jar
kubernetes-model-2.0.0.jar
leveldbjni-all-1.8.jar
libfb303-0.9.3.jar
libthrift-0.9.3.jar
log4j-1.2.17.jar
logging-interceptor-3.8.1.jar
lz4-java-1.4.0.jar
machinist_2.11-0.6.1.jar
macro-compat_2.11-1.1.1.jar
mesos-1.4.0-shaded-protobuf.jar
metrics-core-3.1.5.jar
metrics-graphite-3.1.5.jar
metrics-json-3.1.5.jar
metrics-jvm-3.1.5.jar
minlog-1.3.0.jar
netty-3.9.9.Final.jar
netty-all-4.1.17.Final.jar
objenesis-2.1.jar
okhttp-3.8.1.jar
okio-1.13.0.jar
opencsv-2.3.jar
orc-core-1.4.1-nohive.jar
orc-mapreduce-1.4.1-nohive.jar
oro-2.0.8.jar
osgi-resource-locator-1.0.1.jar
paranamer-2.8.jar
parquet-column-1.8.2.jar
parquet-common-1.8.2.jar
parquet-encoding-1.8.2.jar
parquet-format-2.3.1.jar
parquet-hadoop-1.8.2.jar
parquet-hadoop-bundle-1.6.0.jar
parquet-jackson-1.8.2.jar
protobuf-java-2.5.0.jar
py4j-0.10.6.jar
pyrolite-4.13.jar
scala-compiler-2.11.8.jar
scala-library-2.11.8.jar
scala-parser-combinators_2.11-1.0.4.jar
scala-reflect-2.11.8.jar
scala-xml_2.11-1.0.5.jar
scalap-2.11.8.jar
shapeless_2.11-2.3.2.jar
slf4j-api-1.7.16.jar
slf4j-log4j12-1.7.16.jar
snakeyaml-1.15.jar
snappy-0.2.jar
snappy-java-1.1.2.6.jar
spark-catalyst_2.11-2.3.0.jar
spark-core_2.11-2.3.0.jar
spark-graphx_2.11-2.3.0.jar
spark-hive-thriftserver_2.11-2.3.0.jar
spark-hive_2.11-2.3.0.jar
spark-kubernetes_2.11-2.3.0.jar
spark-kvstore_2.11-2.3.0.jar
spark-launcher_2.11-2.3.0.jar
spark-mesos_2.11-2.3.0.jar
spark-mllib-local_2.11-2.3.0.jar
spark-mllib_2.11-2.3.0.jar
spark-network-common_2.11-2.3.0.jar
spark-network-shuffle_2.11-2.3.0.jar
spark-repl_2.11-2.3.0.jar
spark-sketch_2.11-2.3.0.jar
spark-sql_2.11-2.3.0.jar
spark-streaming_2.11-2.3.0.jar
spark-tags_2.11-2.3.0.jar
spark-unsafe_2.11-2.3.0.jar
spark-yarn_2.11-2.3.0.jar
spire-macros_2.11-0.13.0.jar
spire_2.11-0.13.0.jar
stax-api-1.0-2.jar
stax-api-1.0.1.jar
stream-2.7.0.jar
stringtemplate-3.2.1.jar
super-csv-2.2.0.jar
univocity-parsers-2.5.9.jar
validation-api-1.1.0.Final.jar
xbean-asm5-shaded-4.4.jar
xercesImpl-2.9.1.jar
xmlenc-0.52.jar
xz-1.0.jar
zjsonpatch-0.3.0.jar
zookeeper-3.4.6.jar
zstd-jni-1.3.2-2.jar
