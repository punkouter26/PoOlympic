using System;
using Unity.InferenceEngine;

namespace PoOlympic
{
    /// <summary>
    /// Batch-1 ONNX policy on the Inference Engine CPU backend. Outputs are exactly the graph's `ctrl` (final PD
    /// targets, already clipped) and `action_raw` (fed back as last_action) — C# does no action math.
    /// </summary>
    public sealed class PolicyBrain : IDisposable
    {
        readonly Worker _worker;
        readonly Tensor<float> _input;
        readonly int _obsDim;

        public PolicyBrain(ModelAsset asset, int obsDim)
        {
            _obsDim = obsDim;
            _worker = new Worker(ModelLoader.Load(asset), BackendType.CPU);
            _input = new Tensor<float>(new TensorShape(1, obsDim));
        }

        public void Run(float[] obs, float[] ctrl, float[] actionRaw)
        {
            if (obs.Length != _obsDim) throw new ArgumentException("obs length");
            _input.Upload(obs);
            _worker.Schedule(_input);
            using var c = (_worker.PeekOutput("ctrl") as Tensor<float>).ReadbackAndClone();
            using var a = (_worker.PeekOutput("action_raw") as Tensor<float>).ReadbackAndClone();
            var cArr = c.DownloadToArray();
            var aArr = a.DownloadToArray();
            Array.Copy(cArr, ctrl, ctrl.Length);
            Array.Copy(aArr, actionRaw, actionRaw.Length);
        }

        public void Dispose()
        {
            _input?.Dispose();
            _worker?.Dispose();
        }
    }
}
