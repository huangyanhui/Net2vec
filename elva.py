from sklearn.cluster import KMeans


def loadVec():
    f=open("vec.txt",encoding="utf-8")
    lines=f.readlines()
    lines=lines[1:-1]
    wordvector=[]
    keys=[]
    for line in lines:
        keys.append(line.split(" ")[0])
        wordvector.append(line.split(" ")[1:-1])
    f.close()
    clf = KMeans(n_clusters=200)
    s = clf.fit(wordvector)
    print(s)
    # 获取到所有词向量所属类别
    labels = clf.labels_
    # 把是一类的放入到一个集合

    classCollects={}
    for i in range(len(keys)):
        if labels[i] in classCollects.keys():
            classCollects[labels[i]].append(keys[i])
        else:
            classCollects[labels[i]]=[keys[i]]


    return classCollects




collects=loadVec()
f=open("results.txt",'w',encoding="utf-8")
for i in collects.keys():
    e=collects[i]
    e = ' '.join(map(lambda x: str(x), e))
    f.write("%s\n" % e)
f.close()