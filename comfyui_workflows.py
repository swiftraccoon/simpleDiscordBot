def stoiqo_comfyui_workflow():
    return {
        "4": {
            "inputs": {
            "ckpt_name": "stoiqoNewrealityFLUXSD35_sd35Prealpha.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
            "title": "Load Checkpoint"
            }
        },
        "6": {
            "inputs": {
            "text": "prompt text here",
            "clip": [
                "11",
                0
            ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
            "title": "CLIP Text Encode (Positive Prompt)"
            }
        },
        "8": {
            "inputs": {
            "samples": [
                "294",
                0
            ],
            "vae": [
                "4",
                2
            ]
            },
            "class_type": "VAEDecode",
            "_meta": {
            "title": "VAE Decode"
            }
        },
        "11": {
            "inputs": {
            "clip_name1": "clip_g.safetensors",
            "clip_name2": "clip_l.safetensors",
            "clip_name3": "t5xxl_fp16.safetensors"
            },
            "class_type": "TripleCLIPLoader",
            "_meta": {
            "title": "TripleCLIPLoader"
            }
        },
        "13": {
            "inputs": {
            "shift": 3,
            "model": [
                "4",
                0
            ]
            },
            "class_type": "ModelSamplingSD3",
            "_meta": {
            "title": "ModelSamplingSD3"
            }
        },
        "50": {
            "inputs": {
            "images": [
                "8",
                0
            ]
            },
            "class_type": "PreviewImage",
            "_meta": {
            "title": "Preview Image"
            }
        },
        "67": {
            "inputs": {
            "conditioning": [
                "71",
                0
            ]
            },
            "class_type": "ConditioningZeroOut",
            "_meta": {
            "title": "ConditioningZeroOut"
            }
        },
        "68": {
            "inputs": {
            "start": 0.1,
            "end": 1,
            "conditioning": [
                "67",
                0
            ]
            },
            "class_type": "ConditioningSetTimestepRange",
            "_meta": {
            "title": "ConditioningSetTimestepRange"
            }
        },
        "69": {
            "inputs": {
            "conditioning_1": [
                "68",
                0
            ],
            "conditioning_2": [
                "70",
                0
            ]
            },
            "class_type": "ConditioningCombine",
            "_meta": {
            "title": "Conditioning (Combine)"
            }
        },
        "70": {
            "inputs": {
            "start": 0,
            "end": 0.1,
            "conditioning": [
                "71",
                0
            ]
            },
            "class_type": "ConditioningSetTimestepRange",
            "_meta": {
            "title": "ConditioningSetTimestepRange"
            }
        },
        "71": {
            "inputs": {
            "text": "",
            "clip": [
                "11",
                0
            ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
            "title": "CLIP Text Encode (Negative Prompt)"
            }
        },
        "135": {
            "inputs": {
            "width": 832,
            "height": 1216,
            "batch_size": 1
            },
            "class_type": "EmptySD3LatentImage",
            "_meta": {
            "title": "EmptySD3LatentImage"
            }
        },
        "294": {
            "inputs": {
            "seed": 1086086135934601,
            "steps": 40,
            "cfg": 4.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "sgm_uniform",
            "denoise": 1,
            "model": [
                "13",
                0
            ],
            "positive": [
                "6",
                0
            ],
            "negative": [
                "69",
                0
            ],
            "latent_image": [
                "135",
                0
            ]
            },
            "class_type": "KSampler",
            "_meta": {
            "title": "KSampler"
            }
        }
    }

def sd35medium_comfyui_worfklow():
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 1,
                "denoise": 1,
                "latent_image": [
                    "135",
                    0
                ],
                "model": [
                    "13",
                    0
                ],
                "negative": [
                    "69",
                    0
                ],
                "positive": [
                    "6",
                    0
                ],
                "sampler_name": "dpmpp_2m",
                "scheduler": "sgm_uniform",
                "seed": 867689007911261,
                "steps": 4
            }
        },
        "4": {
            "inputs": {
                "ckpt_name": "stableDiffusion35_largeTurbo.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": "Load Checkpoint"
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": [
                    "11",
                    0
                ],
                "text": "placeholder prompt..."
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": [
                    "3",
                    0
                ],
                "vae": [
                    "4",
                    2
                ]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": [
                    "8",
                    0
                ]
            }
        },
        "11": {
            "class_type": "TripleCLIPLoader",
            "inputs": {
                "clip_name1": "clip_g.safetensors",
                "clip_name2": "clip_l.safetensors",
                "clip_name3": "t5xxl_fp8_e4m3fn_scaled.safetensors"
            }
        },
        "13": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "shift": 3,
                "model": [
                    "4",
                    0
                ]
            }
        },
        "67": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": [
                    "71",
                    0
                ]
            }
        },
        "68": {
            "class_type": "ConditioningSetTimestepRange",
            "inputs": {
                "start": 0.1,
                "end": 1,
                "conditioning": [
                    "67",
                    0
                ]
            }
        },
        "69": {
            "class_type": "ConditioningCombine",
            "inputs": {
                "conditioning_1": [
                    "68",
                    0
                ],
                "conditioning_2": [
                    "70",
                    0
                ]
            }
        },
        "70": {
            "class_type": "ConditioningSetTimestepRange",
            "inputs": {
                "start": 0,
                "end": 0.1,
                "conditioning": [
                    "71",
                    0
                ]
            }
        },
        "71": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "",
                "clip": [
                    "11",
                    0
                ]
            }
        },
        "135": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            }
        }
    }