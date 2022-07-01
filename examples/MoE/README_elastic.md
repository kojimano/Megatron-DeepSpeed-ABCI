DeepSpeed branch: https://github.com/microsoft/DeepSpeed-internal/tree/arpan/elasticity/deepspeed/  
PyTorch version: v1.11.x

Currently, we only support relaunching of training using script method in DeepSpeed (i.e. same script will be ran again when we either scale up or scale down)

bash script: ds_pretrain_gpt_1.3B_dense_elastic.sh

Update dataset path at line number 9


On ITP cluster, there is a need to add hostname (webxt..) to every worker line in "/etc/hosts" file on each worker.

'hosts' file should like something like this

````
# Kubernetes-managed hosts file (host network).
127.0.0.1 localhost

# The following lines are desirable for IPv6 capable hosts
::1 ip6-localhost ip6-loopback
fe00::0 ip6-localnet
ff00::0 ip6-mcastprefix
ff02::1 ip6-allnodes
ff02::2 ip6-allrouters
ff02::3 ip6-allhosts
192.168.0.46    ps-0
192.168.0.46    worker-0 webxt7c7400003X
192.168.0.60    worker-1 webxt7c7400004A
````


Once /etc/hosts file is updated, elastic training can be launched using following command. 

````
bash ds_pretrain_gpt_1.3B_dense_elastic.sh

````

To test fault tolerance, just kill one of the training process on any node. Elastic agent will restart training on all nodes. 

When an elastic agent dies, it is considered a scale down event




