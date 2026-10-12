// Collider view (ColliderXray): the physics primitives, see-through and drawn over the athletes' meshes (ZTest
// Always), brighter at the silhouette so a capsule reads as a round shape. _BaseColor comes per shape from a
// MaterialPropertyBlock (rest colour → loaded colour with the contact force).
Shader "PoOlympic/Xray"
{
    Properties
    {
        _BaseColor("Colour", Color) = (0.35, 0.9, 1, 0.35)
        _Core("Core opacity", Range(0, 1)) = 0.4
    }
    SubShader
    {
        Tags { "RenderType" = "Transparent" "Queue" = "Overlay" "RenderPipeline" = "UniversalPipeline" "IgnoreProjector" = "True" }
        Pass
        {
            Name "Xray"
            Tags { "LightMode" = "UniversalForward" }
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest Always
            Cull Back

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            CBUFFER_START(UnityPerMaterial)
                half4 _BaseColor;
                half _Core;
            CBUFFER_END

            struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; };
            struct Varyings { float4 positionCS : SV_POSITION; float3 positionWS : TEXCOORD0; float3 normalWS : TEXCOORD1; };

            Varyings vert(Attributes v)
            {
                Varyings o;
                o.positionWS = TransformObjectToWorld(v.positionOS.xyz);
                o.positionCS = TransformWorldToHClip(o.positionWS);
                o.normalWS = TransformObjectToWorldNormal(v.normalOS);
                return o;
            }

            half4 frag(Varyings i) : SV_Target
            {
                float facing = saturate(dot(normalize(i.normalWS), normalize(GetWorldSpaceViewDir(i.positionWS))));
                return half4(_BaseColor.rgb, _BaseColor.a * lerp(1.0, _Core, facing));
            }
            ENDHLSL
        }
    }
}
