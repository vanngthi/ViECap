import os, jpype
import phonlp
import py_vncorenlp

# phonlp.download(save_dir='./nlp_models/phonlp')
# py_vncorenlp.download_model(save_dir='./nlp_models/vncorenlp')

jar_dir = os.path.abspath("./nlp_models/vncorenlp")
pho_dir = os.path.abspath("./nlp_models/phonlp")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["CLASSPATH"] = os.path.join(jar_dir, "VnCoreNLP-1.2.jar")

vncorenlp_model = py_vncorenlp.VnCoreNLP(
    save_dir=jar_dir,
    annotators=["wseg"],
    max_heap_size='-Xmx2g'
)

phonlp_model = phonlp.load(pho_dir)

text = "Một vài người đàn ông đang chơi bóng chuyền với một số người xem xung quanh"
segmented = vncorenlp_model.word_segment(text)
print(segmented)

results = phonlp_model.annotate(text=segmented[0])
print(results)            