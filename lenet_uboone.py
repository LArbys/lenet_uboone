import caffe
import numpy as np
import ROOT as rt
import lmdb

caffe.set_mode_gpu()

deploy_prototxt = "/home/taritree/software/caffe/models/lenet_uboone/lenet.prototxt"
test_train_prototxt = "/home/taritree/software/caffe/models/lenet_uboone/lenet_train_test.prototxt"
model = "/home/taritree/software/caffe/models/lenet_uboone/lenet_rmsprop_iter_10000.caffemodel"
weights = "/home/taritree/software/caffe/models/lenet_uboone/lenet_rmsprop_iter_10000.solverstate"
train_data = "/home/taritree/working/larbys/staged_data/uboone_singlep_train.db"
validate_data = "/home/taritree/working/larbys/staged_data/uboone_singlep_validate.db"

prototxt = test_train_prototxt
#prototxt = deploy_prototxt
net = caffe.Net( prototxt, model, caffe.TEST )

lmdb_name = validate_data
lmdb_env = lmdb.open(lmdb_name, readonly=True)
lmdb_txn = lmdb_env.begin()

cursor = lmdb_txn.cursor()

batchsize = 100
nbatches = 10

misslist = []

out = rt.TFile("out_netanalysis.root", "RECREATE" )
herrmat = rt.TH2D("herrmat",";truth label;decision label",4,0,4,4,0,4)
hclassacc = rt.TH1D( "hclassacc", ";truth label;accuracy",4,0,4);
hclassfre = rt.TH1D( "hclassfreq", ";truth label;frequency",4,0,4);
totevents = 0.0
for ibatch in range(0,nbatches):
    keys = []
    for iimg in range(0,batchsize):
        cursor.next()
        (key,raw_datum) = cursor.item()
        #datum = caffe.proto.caffe_pb2.Datum()
        #datum.ParseFromString(raw_datum)
        #feature = caffe.io.datum_to_array(datum)
        #label = datum.label
        #labels.append( label )
        #batch_images.append( feature[:,10:210,10:210] )
        keys.append(key)

    net.forward()
    labels =  net.blobs["label"].data
    scores = net.blobs["ip2"].data
    softmax = net.blobs["loss"].data
    correct = 0.0
    totevents += float( len(scores) )
    for label,score,key in zip(labels,scores,keys):
        #print label,score
        ilabel = int(label)
        decision = np.argmax(score)
        hclassfre.Fill( ilabel )
        if ilabel==decision:
            correct += 1.0
            hclassacc.Fill( ilabel )
        else:
            print "Miss: ",key,label,np.argmax(score)
            misslist.append( {"key":key,"truth_label":int(label),"decision":int(np.argmax(score))} )
            herrmat.Fill( ilabel, decision )
    print "accuracy: ",correct/len(scores)
for miss in misslist:
    print miss

binlabels = {0:"eminus",1:"muminus",2:"proton",3:"pizero"}
for h in [hclassfre,hclassacc,herrmat]:
    for iclass in range(0,4):
        h.GetXaxis().SetBinLabel(iclass+1,binlabels[iclass])
        if h in [herrmat]:
            h.GetYaxis().SetBinLabel(iclass+1,binlabels[iclass])
hclassacc.Divide(hclassfre)
# properly normalize mistake matrix
for iclass in range(0,4):
    tot = 0.0
    for jclass in range(0,4):
        tot += herrmat.GetBinContent( iclass+1, jclass+1 )
    for jclass in range(0,4):
        binval = herrmat.GetBinContent( iclass+1, jclass+1 )
        herrmat.SetBinContent( iclass+1, jclass+1, float(binval)/float(tot) )
    
out.Write()
    
