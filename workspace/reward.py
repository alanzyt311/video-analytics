import csv

data_list=[]
accuracy_normalized=[]
latency_normalized=[]
bandwidth_normalized=[]
timestamp=[]
resolution=[]
fps=[]
qp=[]
accuracy=[]
available_bw=[]
est_bw_config=[]
video_size=[]
latency=[]
objnum=[]
accuracy_obj=[]
bandwidth_obj=[]
latency_obj=[]
cum_accuracy_obj=[]
cum_bandwidth_obj=[]
cum_latency_obj=[]
with open('trafficcam_1.csv', newline='',encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        timestamp.append(row['timestamp'])
        resolution.append(row['resolution'])
        fps.append(row['fps'])
        qp.append(row['qp'])
        accuracy.append(row['accuracy'])
        available_bw.append(row['available_bw'])
        est_bw_config.append(row['est_bw_config'])
        video_size.append(row['video_size'])
        latency.append(row['latency'])
        objnum.append(row['objnum'])
        accuracy_normalized.append(float(row['accuracy'])) #accuracy normalize
csvfile.close()       

'''print(accuracy_normalized)
print(latency)
'''

#bandwidth reward normalized formula: ((available bandwidth-bandwidth usage)+max bandwidth difference)  /(max bandwidth difference - min bandwidth difference)
#Because we only have penalty when bandwidth usage>available bandwidth,the reward is 1 when the usage<available bandwidth. Hence, min bandwidth difference=0
available_bw1=[float(x) for x in available_bw]
est_bw_config1=[float(x) for x in est_bw_config]
min_bandwidth_differ=0
max_bandwidth_differ=50 #这边改最大带宽差 （max 可用带宽和使用带宽的差值）
#get max_bandwidth difference
'''for x,y in zip(available_bw1,est_bw_config1):
        if x-y<0:
            if abs(x-y)>max_bandwidth_differ:
                 max_bandwidth_differ=abs(x-y)'''
#get bandwidth_normalized:
for x,y in zip(available_bw1,est_bw_config1):
     if x-y>=0:
          bandwidth_normalized.append(str((0+max_bandwidth_differ)/(max_bandwidth_differ-min_bandwidth_differ)))
     else:
          bandwidth_normalized.append(str(((x-y)+max_bandwidth_differ)/(max_bandwidth_differ-min_bandwidth_differ)))

#All numbers in original csv are string form,so changed into float.Then found max latency and min latency
latency1=[float(x) for x in latency]
'''min_latency=min(latency1)
max_latency=max(latency1)'''
'''print(min_latency,max_latency)'''
min_latency=1
max_latency=100             #这边改最小时延和最大时延Limitation
#latency reward normalized formula:  (max latency-current latency)/(max latency-min latency)
for actual_latency in latency1:
    latency_normalized.append(str((max_latency-actual_latency)/(max_latency-min_latency)))


#objective function for accuracy, assign weights(accuracy:0.5,bandwidth:0.25,latency:0.25):
for x,y,z in zip(accuracy_normalized,bandwidth_normalized,latency_normalized):
    accuracy_obj.append(0.5*float(x)+0.25*float(y)+0.25*float(z))


cumulative_for_acc=0
for x in range(len(accuracy_obj)):
     if x==0:
        accuracy_obj[x]=0
        cumulative_for_acc+=0
        cum_accuracy_obj.append(0)
     else:
        cumulative_for_acc+=accuracy_obj[x]
        cum_accuracy_obj.append(cumulative_for_acc)

#objective function for bandwidth, assign weights(accuracy:0.25,bandwidth:0.5,latency:0.25):
for x,y,z in zip(accuracy_normalized,bandwidth_normalized,latency_normalized):
    bandwidth_obj.append(0.25*float(x)+0.5*float(y)+0.25*float(z))


cumulative_for_bw=0
for x in range(len(bandwidth_obj)):
     if x==0:
        bandwidth_obj[x]=0
        cumulative_for_bw+=0
        cum_bandwidth_obj.append(0)
     else:
        cumulative_for_bw+=bandwidth_obj[x]
        cum_bandwidth_obj.append(cumulative_for_bw)

#objective function for latency, assign weights(accuracy:0.25,bandwidth:0.25,latency:0.5):
for x,y,z in zip(accuracy_normalized,bandwidth_normalized,latency_normalized):
    latency_obj.append(0.25*float(x)+0.25*float(y)+0.5*float(z))


cumulative_for_latency=0
for x in range(len(latency_obj)): 
     if x==0:
        latency_obj[x]=0
        cumulative_for_latency+=0
        cum_latency_obj.append(0)
     else:
        cumulative_for_latency+=latency_obj[x]
        cum_latency_obj.append(cumulative_for_latency)



'''print(max_bandwidth_differ)
print(available_bw1)
'''


for timestamp,resolution,fps,qp,accuracy,available_bw,est_bw_config,video_size,latency,objnum ,accuracy_normalized,bandwidth_normalized,latency_normalized,accuracy_obj,cum_accuracy_obj,bandwidth_obj,cum_bandwidth_obj,latency_obj,cum_latency_obj in zip(timestamp,
resolution,fps,qp,accuracy,available_bw,est_bw_config,video_size,latency,objnum,accuracy_normalized,bandwidth_normalized,latency_normalized,accuracy_obj,
cum_accuracy_obj,bandwidth_obj,cum_bandwidth_obj,latency_obj,cum_latency_obj):
    data_list.append({'timestamp':timestamp,'resolution':resolution,'fps':fps,'qp':qp,'accuracy':accuracy,'available_bw':available_bw,
                      'est_bw_config':est_bw_config,'video_size':video_size,'latency':latency,'objnum':objnum,'accuracy_normalized':accuracy_normalized,
                      'bandwidth_normalized':bandwidth_normalized,'latency_normalized':latency_normalized,'accuracy_obj':accuracy_obj,'cum_accuracy_obj':cum_accuracy_obj,
                      'bandwidth_obj':bandwidth_obj,'cum_bandwidth_obj':cum_bandwidth_obj,'latency_obj':latency_obj,'cum_latency_obj':cum_latency_obj})

fname = 'trafficcam_1.csv'
with open(fname,"w",encoding="utf8",newline="") as outputfile:
    writer=csv.DictWriter(outputfile,fieldnames=['timestamp','resolution','fps','qp','accuracy','available_bw','est_bw_config','video_size','latency','objnum' ,
                                                 'accuracy_normalized','bandwidth_normalized','latency_normalized','accuracy_obj','cum_accuracy_obj','bandwidth_obj','cum_bandwidth_obj','latency_obj','cum_latency_obj'])
    writer.writeheader()
    writer.writerows( )
