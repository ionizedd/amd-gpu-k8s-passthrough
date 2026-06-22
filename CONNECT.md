# Connecting to the GPU Cluster

## What you need

- [Tailscale](https://tailscale.com) installed and logged into the `Fleabag515` tailnet
- [kubectl](https://kubernetes.io/docs/tasks/tools/) installed
- The `fleabag-kubeconfig.yaml` file (sent to you separately — keep it private!)

## Setup (one time)

```bash
mkdir -p ~/.kube
cp fleabag-kubeconfig.yaml ~/.kube/fleabag-config
export KUBECONFIG=~/.kube/fleabag-config

# Verify you can reach the cluster
kubectl get pods -n gpu-jobs
```

Add the export to your `~/.bashrc` to make it permanent.

## Submitting a training job

Create a `my-job.yaml` (based on `gpu-pod.yaml` in this repo), then:

```bash
# Run your training
kubectl apply -f my-job.yaml

# Watch the logs live
kubectl logs -f my-training-pod -n gpu-jobs

# Clean up when done
kubectl delete pod my-training-pod -n gpu-jobs
```

## Mounting your code

The easiest way to get your script into the pod is to add a ConfigMap:

```bash
kubectl create configmap my-script --from-file=train.py -n gpu-jobs
```

Then mount it in your pod manifest:
```yaml
volumes:
  - name: script
    configMap:
      name: my-script
containers:
  - volumeMounts:
    - name: script
      mountPath: /workspace
```

## Permissions

Your token can only create/delete pods and jobs in the `gpu-jobs` namespace. 
It cannot touch any other part of the cluster.

## Notes

- The cluster runs on an **AMD RX 9070 XT** (16 GB VRAM) via ROCm 6.4
- Your pod needs the same volume mounts as `gpu-pod.yaml` to reach the GPU
- The token expires in **1 year** — ask for a renewal when needed
