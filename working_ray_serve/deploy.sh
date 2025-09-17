set -e

# 1. Build & tag
docker build -t localhost:5000/ray-triton-app:v1 .

# 2. Push to local registry
docker push localhost:5000/ray-triton-app:v1

# 3. Apply the RayService
kubectl apply -n rayserve -f raycluster.yaml

# 4. Verify
kubectl -n rayserve rollout status rayservice/image-serve
kubectl -n rayserve get pods -l ray.io/cluster=image-serve