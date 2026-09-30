// Arena light beams (GFX idea 4): additive cones under the field spots (build_showcase.py beams, UV v = 0 at the lens,
// 1 at the far end). Bright core facing the viewer, soft silhouette (fresnel), fade along the beam, fade near the
// camera (a camera inside a cone must not fog the whole frame) and in fog. _PoBeamDim (global, 0 = full, 1 = off) is raised by
// the quality scaler (PerfOverlay) on a hot phone.
Shader "PoOlympic/LightBeam"
{
    Properties
    {
        [HDR] _Color("Colour", Color) = (1, 0.97, 0.9, 1)
        _Intensity("Intensity", Range(0, 1)) = 0.06
        _EdgePower("Edge softness", Range(0.5, 6)) = 2.2
        _NearFade("Near fade distance (m)", Float) = 6
    }
    SubShader
    {
        Tags { "RenderType" = "Transparent" "Queue" = "Transparent+10" "RenderPipeline" = "UniversalPipeline" "IgnoreProjector" = "True" }
        Pass
        {
            Name "Beam"
            Tags { "LightMode" = "UniversalForward" }
            Blend One One
            ZWrite Off
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fog
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            CBUFFER_START(UnityPerMaterial)
                half4 _Color;
                half _Intensity, _EdgePower, _NearFade;
            CBUFFER_END
            float _PoBeamDim;

            struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; float2 uv : TEXCOORD0; };
            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float2 uv : TEXCOORD2;
                float fog : TEXCOORD3;
            };

            Varyings vert(Attributes v)
            {
                Varyings o;
                o.positionWS = TransformObjectToWorld(v.positionOS.xyz);
                o.positionCS = TransformWorldToHClip(o.positionWS);
                o.normalWS = TransformObjectToWorldNormal(v.normalOS);
                o.uv = v.uv;
                o.fog = ComputeFogFactor(o.positionCS.z);
                return o;
            }

            half4 frag(Varyings i) : SV_Target
            {
                float3 view = GetWorldSpaceViewDir(i.positionWS);
                float dist = length(view);
                float facing = abs(dot(normalize(i.normalWS), view / max(dist, 1e-4)));
                float core = pow(saturate(facing), _EdgePower);
                float along = (1.0 - i.uv.y) * (1.0 - i.uv.y) * smoothstep(0.0, 0.08, i.uv.y);
                float near = saturate((dist - 1.0) / max(_NearFade, 0.01));
                float scale = saturate(1.0 - _PoBeamDim);
                half3 c = _Color.rgb * (_Intensity * core * along * near * scale);
                c *= ComputeFogIntensity(i.fog);                      // additive: fade out into the fog
                return half4(c, 0);
            }
            ENDHLSL
        }
    }
}
