"use strict";

// ── Constants ──────────────────────────────────────────────────
const LS_KEY          = "aiml_platform_progress";
const LS_KEY_PROJECTS = "aiml_platform_projects_progress";
const LOAD_START      = Date.now();
const MIN_LOAD_MS     = 1400;

const MODULE_META = [
  { file: "01-math.md",           label: "Math for ML",               tag: "01", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><path d="M10.315 4.876L6.3048 2.8517l-4.401 2.1965 4.1186 2.0683zm1.8381.9277l4.2045 2.1223-4.3622 2.1906-4.125-2.0718zm5.6153-2.9213l4.3193 2.1658-3.863 1.9402-4.2131-2.1252zm-1.859-.9329L12.021 0 8.1742 1.9193l4.0068 2.0208zm-3.0401 16.7443V24l4.7107-2.3507-.0053-5.3085zm4.7037-4.2057l-.0052-5.2528-4.6985 2.3356v5.2546zm5.6553-.9845v5.327l-4.0178 2.0052-.0029-5.3028zm0-1.8626V6.4214l-4.0253 2.001.0034 5.2633zM11.2062 11.571L8.0333 9.9756v6.895s-3.8804-8.2564-4.2399-8.998c-.0463-.0957-.2371-.2007-.2858-.2262C2.8118 7.2812.773 6.2485.773 6.2485V18.43l2.8204 1.5076v-6.3674s3.8392 7.3775 3.878 7.458c.0389.0807.4245.8582.8362 1.1314.5485.363 2.8992 1.7766 2.8992 1.7766z"/></svg>` },
  { file: "02-ml-basics.md",      label: "ML Basics → Advanced",      tag: "02", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><path d="M15.601 5.53c-1.91.035-3.981.91-5.63 2.56-2.93 2.93-2.083 8.53-1.088 9.525.805.804 6.595 1.843 9.526-1.088a9.74 9.74 0 0 0 .584-.643c.043-.292.205-.66.489-1.106a1.848 1.848 0 0 1-.537.176c-.144.265-.37.55-.676.855-.354.335-.607.554-.76.656a.795.795 0 0 1-.437.152c-.35 0-.514-.308-.494-.924-.22.316-.425.549-.612.7a.914.914 0 0 1-.578.224c-.194 0-.36-.09-.496-.273a1.03 1.03 0 0 1-.193-.507 4.016 4.016 0 0 1-.726.583c-.224.132-.47.197-.74.197-.3 0-.543-.096-.727-.288a.978.978 0 0 1-.257-.524v.004c-.3.276-.564.48-.79.611a1.295 1.295 0 0 1-.649.197.693.693 0 0 1-.571-.275c-.145-.183-.218-.43-.218-.739 0-.464.101-1.02.302-1.67.201-.65.445-1.25.733-1.797l.842-.312a.21.21 0 0 1 .06-.013c.063 0 .116.047.157.14.04.095.061.221.061.38 0 .451-.104.888-.312 1.31-.207.422-.532.873-.974 1.352-.018.23-.027.388-.027.474 0 .193.036.345.106.458.071.113.165.169.282.169a.71.71 0 0 0 .382-.13c.132-.084.333-.26.602-.523.028-.418.187-.798.482-1.142.324-.38.685-.569 1.08-.569.206 0 .37.054.494.16a.524.524 0 0 1 .186.417c0 .458-.486.829-1.459 1.114.088.43.32.646.693.646a.807.807 0 0 0 .417-.117c.129-.076.321-.243.575-.497.032-.252.118-.495.259-.728.182-.3.416-.544.701-.73.285-.185.537-.278.756-.278.276 0 .47.127.58.381l.677-.374h.186l-.292.971c-.15.488-.226.823-.226 1.004 0 .19.067.285.202.285.086 0 .181-.045.285-.137.104-.092.25-.232.437-.42v.001c.143-.155.274-.32.392-.494-.19-.084-.285-.21-.285-.375 0-.17.058-.352.174-.545.116-.194.275-.29.479-.29.172 0 .258.088.258.265 0 .139-.05.338-.149.596.367-.04.687-.32.961-.842l.228-.01c1.059-2.438.828-5.075-.83-6.732-1.019-1.02-2.408-1.5-3.895-1.471zm4.725 8.203a8.938 8.938 0 0 1-1.333 2.151 1.09 1.09 0 0 0-.012.147c0 .168.047.309.14.423.092.113.206.17.34.17.296 0 .714-.264 1.254-.787-.001.04-.003.08-.003.121 0 .146.012.368.036.666l.733-.172c0-.2.003-.357.01-.474.01-.157.033-.33.066-.517.02-.11.07-.216.152-.315l.186-.216a5.276 5.276 0 0 1 .378-.397c.062-.055.116-.099.162-.13a.26.26 0 0 1 .123-.046c.055 0 .083.035.083.106 0 .07-.052.236-.156.497-.194.486-.292.848-.292 1.084 0 .175.046.314.136.418a.45.45 0 0 0 .358.155c.365 0 .803-.269 1.313-.808v-.381c-.361.426-.623.64-.784.64-.109 0-.163-.067-.163-.2 0-.1.065-.316.195-.65.19-.486.285-.836.285-1.048a.464.464 0 0 0-.112-.319.36.36 0 0 0-.282-.127c-.165 0-.354.077-.567.233-.213.156-.5.436-.863.84.053-.262.165-.622.335-1.08l-.809.156a6.54 6.54 0 0 0-.399 1.074c-.04.156-.07.316-.092.48a7.447 7.447 0 0 1-.49.45.38.38 0 0 1-.229.08.208.208 0 0 1-.174-.082.352.352 0 0 1-.064-.222c0-.1.019-.214.056-.343.038-.13.12-.373.249-.731l.308-.849zm-17.21-2.927c-.863-.016-1.67.263-2.261.854-1.352 1.352-1.07 3.827.631 5.527 1.7 1.701 4.95 1.21 5.527.632.467-.466 1.07-3.827-.631-5.527-.957-.957-2.158-1.465-3.267-1.486zm12.285.358h.166v.21H15.4zm.427 0h.166v.865l.46-.455h.195l-.364.362.428.684h-.198l-.357-.575-.164.166v.41h-.166zm1.016 0h.166v.21h-.166zm.481.122h.166v.288h.172v.135h-.172v.717c0 .037.006.062.02.075.012.013.037.02.074.02a.23.23 0 0 0 .078-.01v.141a.802.802 0 0 1-.136.014.23.23 0 0 1-.15-.043.15.15 0 0 1-.052-.123v-.79h-.141v-.136h.141zm-3.562.258c.081 0 .15.012.207.038.057.024.1.061.13.11s.045.106.045.173h-.176c-.006-.111-.075-.167-.208-.167a.285.285 0 0 0-.164.041.134.134 0 0 0-.06.117c0 .035.015.065.045.088.03.024.08.044.15.06l.16.039a.47.47 0 0 1 .224.105c.047.046.07.108.07.186a.3.3 0 0 1-.052.175.327.327 0 0 1-.152.116.585.585 0 0 1-.226.041c-.136 0-.24-.03-.309-.088-.069-.059-.105-.149-.109-.269h.176c.004.037.01.065.017.084a.166.166 0 0 0 .034.054c.044.043.112.065.204.065a.31.31 0 0 0 .177-.045.139.139 0 0 0 .067-.119.116.116 0 0 0-.038-.09.287.287 0 0 0-.124-.055l-.156-.038a1.248 1.248 0 0 1-.159-.05.359.359 0 0 1-.098-.061.22.22 0 0 1-.058-.083.32.32 0 0 1-.016-.108c0-.096.036-.174.109-.232a.45.45 0 0 1 .29-.087zm1.035 0a.46.46 0 0 1 .202.043.351.351 0 0 1 .187.212.577.577 0 0 1 .023.126h-.168a.256.256 0 0 0-.078-.168.242.242 0 0 0-.17-.06.248.248 0 0 0-.155.05.306.306 0 0 0-.1.144.662.662 0 0 0-.034.224.58.58 0 0 0 .035.214.299.299 0 0 0 .101.135.261.261 0 0 0 .157.048c.142 0 .227-.084.256-.252h.167a.519.519 0 0 1-.065.22.35.35 0 0 1-.146.138.464.464 0 0 1-.216.048.448.448 0 0 1-.246-.066.441.441 0 0 1-.161-.192.703.703 0 0 1-.057-.293c0-.085.01-.163.032-.233a.522.522 0 0 1 .095-.182.403.403 0 0 1 .15-.117.453.453 0 0 1 .191-.04zm.603.03h.166v1.046H15.4zm1.443 0h.166v1.046h-.166zm-5.05.618c-.08 0-.2.204-.356.611-.155.407-.308.977-.459 1.71.281-.312.509-.662.683-1.05.175-.387.262-.72.262-.999a.455.455 0 0 0-.036-.197c-.025-.05-.056-.075-.093-.075zm4.662 1.797c-.221 0-.431.188-.629.563-.197.376-.296.722-.296 1.038 0 .12.029.216.088.29a.273.273 0 0 0 .223.111c.221 0 .43-.188.625-.565.196-.377.294-.725.294-1.043a.457.457 0 0 0-.083-.29.269.269 0 0 0-.222-.104zm-2.848.007c-.146 0-.285.11-.417.333-.133.222-.2.51-.2.866.566-.159.849-.452.849-.881 0-.212-.077-.318-.232-.318Z"/></svg>` },
  { file: "03-databases.md",      label: "Databases & Vector DBs",    tag: "03", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><path d="M13.394 0C8.683 0 4.609 2.716 2.644 6.667h15.641a4.77 4.77 0 0 0 3.073-1.11c.446-.375.864-.785 1.247-1.243l.001-.002A11.974 11.974 0 0 0 13.394 0zM1.804 8.889a12.009 12.009 0 0 0 0 6.222h14.7a3.111 3.111 0 1 0 0-6.222zm.84 8.444C4.61 21.283 8.684 24 13.395 24c3.701 0 7.011-1.677 9.212-4.312l-.001-.002a9.958 9.958 0 0 0-1.247-1.243 4.77 4.77 0 0 0-3.073-1.11z"/></svg>` },
  { file: "04-backend.md",        label: "Backend with Flask",        tag: "04", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><path d="M10.773 2.878c-.013 1.434.322 4.624.445 5.734l-8.558 3.83c-.56-.959-.98-2.304-1.237-3.38l-.06.027c-.205.09-.406.053-.494-.088l-.011-.018-.82-1.506c-.058-.105-.05-.252.024-.392a.78.78 0 0 1 .358-.331l9.824-4.207c.146-.064.299-.063.4.004.106.062.127.128.13.327Zm.68 7c.523 1.97.675 2.412.832 2.818l-7.263 3.7a19.35 19.35 0 0 1-1.81-2.83l8.24-3.689Zm12.432 8.786h.003c.283.402-.047.657-.153.698l-.947.37c.037.125.035.319-.217.414l-.736.287c-.229.09-.398-.059-.42-.2l-.025-.125c-4.427 1.784-7.94 1.685-10.696.647-1.981-.745-3.576-1.983-4.846-3.379l6.948-3.54c.721 1.431 1.586 2.454 2.509 3.178 2.086 1.638 4.415 1.712 5.793 1.563l-.047-.233c-.015-.077.007-.135.086-.165l.734-.288a.302.302 0 0 1 .342.086l.748-.288a.306.306 0 0 1 .341.086l.583.89Z"/></svg>` },
  { file: "05-deep-learning.md",  label: "Deep Learning",             tag: "05", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><path d="M12.005 0L4.952 7.053a9.865 9.865 0 000 14.022 9.866 9.866 0 0014.022 0c3.984-3.9 3.986-10.205.085-14.023l-1.744 1.743c2.904 2.905 2.904 7.634 0 10.538s-7.634 2.904-10.538 0-2.904-7.634 0-10.538l4.647-4.646.582-.665zm3.568 3.899a1.327 1.327 0 00-1.327 1.327 1.327 1.327 0 001.327 1.328A1.327 1.327 0 0016.9 5.226 1.327 1.327 0 0015.573 3.9z"/></svg>` },
  { file: "06-genai-core.md",     label: "GenAI Core",                tag: "06", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z"/></svg>` },
  { file: "07-transformers.md",   label: "Transformers from Scratch", tag: "07", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><path d="M1.292 5.856L11.54 0v24l-4.095-2.378V7.603l-6.168 3.564.015-5.31zm21.43 5.311l-.014-5.31L12.46 0v24l4.095-2.378V14.87l3.092 1.788-.018-4.618-3.074-1.756V7.603l6.168 3.564z"/></svg>` },
  { file: "08-rag.md",            label: "RAG Systems",               tag: "08", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><path d="M13.796 0a6.93 6.93 0 0 0-4.91 2.019L5.451 5.455l3.273 3.27 3.432-3.432a2.284 2.284 0 0 1 3.277 0 2.28 2.28 0 0 1 0 3.275L12 12.001l3.273 3.273 3.433-3.435c2.692-2.692 2.692-7.127 0-9.82A6.92 6.92 0 0 0 13.796 0m-5.07 8.728-3.433 3.434c-2.692 2.693-2.692 7.126 0 9.819A6.92 6.92 0 0 0 10.203 24a6.93 6.93 0 0 0 4.911-2.02l3.432-3.432-3.271-3.272-3.433 3.433a2.284 2.284 0 0 1-3.277 0 2.28 2.28 0 0 1 0-3.276L12 12z"/></svg>` },
  { file: "09-finetuning.md",     label: "Fine-Tuning & LoRA",        tag: "09", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><rect x="2" y="5" width="20" height="2" rx="1"/><circle cx="16" cy="6" r="3.5"/><rect x="2" y="12" width="20" height="2" rx="1"/><circle cx="10" cy="13" r="3.5"/><rect x="2" y="19" width="20" height="2" rx="1"/><circle cx="6" cy="20" r="3.5"/></svg>` },
  { file: "10-agents.md",         label: "LLM Agents & Tool Use",     tag: "10", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><rect x="7" y="7" width="10" height="10" rx="1.5"/><rect x="10" y="3" width="1.5" height="4"/><rect x="12.5" y="3" width="1.5" height="4"/><rect x="10" y="17" width="1.5" height="4"/><rect x="12.5" y="17" width="1.5" height="4"/><rect x="3" y="10" width="4" height="1.5"/><rect x="3" y="12.5" width="4" height="1.5"/><rect x="17" y="10" width="4" height="1.5"/><rect x="17" y="12.5" width="4" height="1.5"/></svg>` },
  { file: "11-deployment.md",     label: "Deployment & Production",   tag: "11", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><path d="M13.983 11.078h2.119a.186.186 0 00.186-.185V9.006a.186.186 0 00-.186-.186h-2.119a.185.185 0 00-.185.185v1.888c0 .102.083.185.185.185m-2.954-5.43h2.118a.186.186 0 00.186-.186V3.574a.186.186 0 00-.186-.185h-2.118a.185.185 0 00-.185.185v1.888c0 .102.082.185.185.185m0 2.716h2.118a.187.187 0 00.186-.186V6.29a.186.186 0 00-.186-.185h-2.118a.185.185 0 00-.185.185v1.887c0 .102.082.185.185.186m-2.93 0h2.12a.186.186 0 00.184-.186V6.29a.185.185 0 00-.185-.185H8.1a.185.185 0 00-.185.185v1.887c0 .102.083.185.185.186m-2.964 0h2.119a.186.186 0 00.185-.186V6.29a.185.185 0 00-.185-.185H5.136a.186.186 0 00-.186.185v1.887c0 .102.084.185.186.186m5.893 2.715h2.118a.186.186 0 00.186-.185V9.006a.186.186 0 00-.186-.186h-2.118a.185.185 0 00-.185.185v1.888c0 .102.082.185.185.185m-2.93 0h2.12a.185.185 0 00.184-.185V9.006a.185.185 0 00-.184-.186h-2.12a.185.185 0 00-.184.185v1.888c0 .102.083.185.185.185m-2.964 0h2.119a.185.185 0 00.185-.185V9.006a.185.185 0 00-.184-.186h-2.12a.186.186 0 00-.186.186v1.887c0 .102.084.185.186.185m-2.92 0h2.12a.185.185 0 00.184-.185V9.006a.185.185 0 00-.184-.186h-2.12a.185.185 0 00-.184.185v1.888c0 .102.082.185.185.185M23.763 9.89c-.065-.051-.672-.51-1.954-.51-.338.001-.676.03-1.01.087-.248-1.7-1.653-2.53-1.716-2.566l-.344-.199-.226.327c-.284.438-.49.922-.612 1.43-.23.97-.09 1.882.403 2.661-.595.332-1.55.413-1.744.42H.751a.751.751 0 00-.75.748 11.376 11.376 0 00.692 4.062c.545 1.428 1.355 2.48 2.41 3.124 1.18.723 3.1 1.137 5.275 1.137.983.003 1.963-.086 2.93-.266a12.248 12.248 0 003.823-1.389c.98-.567 1.86-1.288 2.61-2.136 1.252-1.418 1.998-2.997 2.553-4.4h.221c1.372 0 2.215-.549 2.68-1.009.309-.293.55-.65.707-1.046l.098-.288Z"/></svg>` },
  { file: "12-rlhf.md",           label: "RLHF & Alignment",          tag: "12", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17.4l-6.2 3.5 2.4-7.4L2 9.4h7.6z"/></svg>` },
  { file: "13-multimodal.md",     label: "Multimodal Models",         tag: "13", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm8-9h-3.17L15 4H9L7.17 6H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2zm-8 13a5 5 0 1 1 0-10 5 5 0 0 1 0 10z"/></svg>` },
  { file: "14-frontend.md",       label: "Frontend (React+Tailwind)", tag: "14", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor"><path d="M14.23 12.004a2.236 2.236 0 0 1-2.235 2.236 2.236 2.236 0 0 1-2.236-2.236 2.236 2.236 0 0 1 2.235-2.236 2.236 2.236 0 0 1 2.236 2.236zm2.648-10.69c-1.346 0-3.107.96-4.888 2.622-1.78-1.653-3.542-2.602-4.887-2.602-.41 0-.783.093-1.106.278-1.375.793-1.683 3.264-.973 6.365C1.98 8.917 0 10.42 0 12.004c0 1.59 1.99 3.097 5.043 4.03-.704 3.113-.39 5.588.988 6.38.32.187.69.275 1.102.275 1.345 0 3.107-.96 4.888-2.624 1.78 1.654 3.542 2.603 4.887 2.603.41 0 .783-.09 1.106-.275 1.374-.792 1.683-3.263.973-6.365C22.02 15.096 24 13.59 24 12.004c0-1.59-1.99-3.097-5.043-4.032.704-3.11.39-5.587-.988-6.38-.318-.184-.688-.277-1.092-.278zm-.005 1.09v.006c.225 0 .406.044.558.127.666.382.955 1.835.73 3.704-.054.46-.142.945-.25 1.44-.96-.236-2.006-.417-3.107-.534-.66-.905-1.345-1.727-2.035-2.447 1.592-1.48 3.087-2.292 4.105-2.295zm-9.77.02c1.012 0 2.514.808 4.11 2.28-.686.72-1.37 1.537-2.02 2.442-1.107.117-2.154.298-3.113.538-.112-.49-.195-.964-.254-1.42-.23-1.868.054-3.32.714-3.707.19-.09.4-.127.563-.132zm4.882 3.05c.455.468.91.992 1.36 1.564-.44-.02-.89-.034-1.345-.034-.46 0-.915.01-1.36.034.44-.572.895-1.096 1.345-1.565zM12 8.1c.74 0 1.477.034 2.202.093.406.582.802 1.203 1.183 1.86.372.64.71 1.29 1.018 1.946-.308.655-.646 1.31-1.013 1.95-.38.66-.773 1.288-1.18 1.87-.728.063-1.466.098-2.21.098-.74 0-1.477-.035-2.202-.093-.406-.582-.802-1.204-1.183-1.86-.372-.64-.71-1.29-1.018-1.946.303-.657.646-1.313 1.013-1.954.38-.66.773-1.286 1.18-1.868.728-.064 1.466-.098 2.21-.098zm-3.635.254c-.24.377-.48.763-.704 1.16-.225.39-.435.782-.635 1.174-.265-.656-.49-1.31-.676-1.947.64-.15 1.315-.283 2.015-.386zm7.26 0c.695.103 1.365.23 2.006.387-.18.632-.405 1.282-.66 1.933-.2-.39-.41-.783-.64-1.174-.225-.392-.465-.774-.705-1.146zm3.063.675c.484.15.944.317 1.375.498 1.732.74 2.852 1.708 2.852 2.476-.005.768-1.125 1.74-2.857 2.475-.42.18-.88.342-1.355.493-.28-.958-.646-1.956-1.1-2.98.45-1.017.81-2.01 1.085-2.964zm-13.395.004c.278.96.645 1.957 1.1 2.98-.45 1.017-.812 2.01-1.086 2.964-.484-.15-.944-.318-1.37-.5-1.732-.737-2.852-1.706-2.852-2.474 0-.768 1.12-1.742 2.852-2.476.42-.18.88-.342 1.356-.494zm11.678 4.28c.265.657.49 1.312.676 1.948-.64.157-1.316.29-2.016.39.24-.375.48-.762.705-1.158.225-.39.435-.788.636-1.18zm-9.945.02c.2.392.41.783.64 1.175.23.39.465.772.705 1.143-.695-.102-1.365-.23-2.006-.386.18-.63.406-1.282.66-1.933zM17.92 16.32c.112.493.2.968.254 1.423.23 1.868-.054 3.32-.714 3.708-.147.09-.338.128-.563.128-1.012 0-2.514-.807-4.11-2.28.686-.72 1.37-1.536 2.02-2.44 1.107-.118 2.154-.3 3.113-.54zm-11.83.01c.96.234 2.006.415 3.107.532.66.905 1.345 1.727 2.035 2.446-1.595 1.483-3.092 2.295-4.11 2.295-.22-.005-.406-.05-.553-.132-.666-.38-.955-1.834-.73-3.703.054-.46.142-.944.25-1.438zm4.56.64c.44.02.89.034 1.345.034.46 0 .915-.01 1.36-.034-.44.572-.895 1.095-1.345 1.565-.455-.47-.91-.993-1.36-1.565z"/></svg>` },
];

const CODE_META = [
  { file: "01-math/calculus_demo.py",          label: "calculus_demo.py",          module: "01" },
  { file: "01-math/matrix_ops.py",             label: "matrix_ops.py",             module: "01" },
  { file: "01-math/probability.py",            label: "probability.py",            module: "01" },
  { file: "01-math/vectors.py",                label: "vectors.py",                module: "01" },
  { file: "02-ml/clustering.py",               label: "clustering.py",             module: "02" },
  { file: "02-ml/decision_tree.py",            label: "decision_tree.py",          module: "02" },
  { file: "02-ml/evaluation.py",               label: "evaluation.py",             module: "02" },
  { file: "02-ml/gradient_boosting.py",        label: "gradient_boosting.py",      module: "02" },
  { file: "02-ml/linear_regression.py",        label: "linear_regression.py",      module: "02" },
  { file: "02-ml/logistic_regression.py",      label: "logistic_regression.py",    module: "02" },
  { file: "02-ml/pca.py",                      label: "pca.py",                    module: "02" },
  { file: "02-ml/random_forest.py",            label: "random_forest.py",          module: "02" },
  { file: "02-ml/svm.py",                      label: "svm.py",                    module: "02" },
  { file: "03-databases/chroma_demo.py",       label: "chroma_demo.py",            module: "03" },
  { file: "03-databases/faiss_demo.py",        label: "faiss_demo.py",             module: "03" },
  { file: "03-databases/nosql_patterns.py",    label: "nosql_patterns.py",         module: "03" },
  { file: "03-databases/pinecone_demo.py",     label: "pinecone_demo.py",          module: "03" },
  { file: "03-databases/sql_basics.py",        label: "sql_basics.py",             module: "03" },
  { file: "04-backend/app.py",                 label: "app.py",                    module: "04" },
  { file: "04-backend/async_tasks.py",         label: "async_tasks.py",            module: "04" },
  { file: "04-backend/middleware.py",          label: "middleware.py",             module: "04" },
  { file: "04-backend/ml_serving.py",          label: "ml_serving.py",             module: "04" },
  { file: "05-deep-learning/mlflow_demo.py",   label: "mlflow_demo.py",            module: "05" },
  { file: "05-deep-learning/monitoring.py",    label: "monitoring.py",             module: "05" },
  { file: "05-deep-learning/nn_numpy.py",      label: "nn_numpy.py",               module: "05" },
  { file: "05-deep-learning/onnx_export.py",   label: "onnx_export.py",            module: "05" },
  { file: "05-deep-learning/optimizers.py",    label: "optimizers.py",             module: "05" },
  { file: "06-genai/attention.py",             label: "attention.py",              module: "06" },
  { file: "06-genai/kv_cache.py",              label: "kv_cache.py",               module: "06" },
  { file: "06-genai/multihead_attention.py",   label: "multihead_attention.py",    module: "06" },
  { file: "06-genai/positional_encoding.py",   label: "positional_encoding.py",    module: "06" },
  { file: "06-genai/word2vec.py",              label: "word2vec.py",               module: "06" },
  { file: "07-transformer/model.cpp",          label: "model.cpp",                 module: "07" },
  { file: "07-transformer/model.py",           label: "model.py",                  module: "07" },
  { file: "07-transformer/model_numpy.py",     label: "model_numpy.py",            module: "07" },
  { file: "07-transformer/tokenizer.py",       label: "tokenizer.py",              module: "07" },
  { file: "07-transformer/train.py",           label: "train.py",                  module: "07" },
  { file: "08-rag/app.py",                     label: "app.py",                    module: "08" },
  { file: "08-rag/embed_store.py",             label: "embed_store.py",            module: "08" },
  { file: "08-rag/evaluate.py",                label: "evaluate.py",               module: "08" },
  { file: "08-rag/generator.py",               label: "generator.py",              module: "08" },
  { file: "08-rag/ingest.py",                  label: "ingest.py",                 module: "08" },
  { file: "08-rag/retriever.py",               label: "retriever.py",              module: "08" },
  { file: "09-finetuning/evaluate.py",         label: "evaluate.py",               module: "09" },
  { file: "09-finetuning/lora_theory.py",      label: "lora_theory.py",            module: "09" },
  { file: "09-finetuning/merge_push.py",       label: "merge_push.py",             module: "09" },
  { file: "09-finetuning/prepare_dataset.py",  label: "prepare_dataset.py",        module: "09" },
  { file: "09-finetuning/train_lora.py",       label: "train_lora.py",             module: "09" },
  { file: "09-finetuning/train_qlora.py",      label: "train_qlora.py",            module: "09" },
  { file: "10-agents/agent_eval.py",           label: "agent_eval.py",             module: "10" },
  { file: "10-agents/agent_memory.py",         label: "agent_memory.py",           module: "10" },
  { file: "10-agents/react_agent.py",          label: "react_agent.py",            module: "10" },
  { file: "10-agents/tool_calling.py",         label: "tool_calling.py",           module: "10" },
  { file: "11-deployment/ab_serving.py",       label: "ab_serving.py",             module: "11" },
  { file: "11-deployment/health_check.py",     label: "health_check.py",           module: "11" },
  { file: "11-deployment/onnx_export.py",      label: "onnx_export.py",            module: "11" },
  { file: "11-deployment/quantize.py",         label: "quantize.py",               module: "11" },
  { file: "12-rlhf/dpo.py",                    label: "dpo.py",                    module: "12" },
  { file: "12-rlhf/evaluate_alignment.py",     label: "evaluate_alignment.py",     module: "12" },
  { file: "12-rlhf/ppo_scratch.py",            label: "ppo_scratch.py",            module: "12" },
  { file: "12-rlhf/reward_model.py",           label: "reward_model.py",           module: "12" },
  { file: "13-multimodal/captioning.py",       label: "captioning.py",             module: "13" },
  { file: "13-multimodal/clip_scratch.py",     label: "clip_scratch.py",           module: "13" },
  { file: "13-multimodal/vit_patch.py",        label: "vit_patch.py",              module: "13" },
  { file: "13-multimodal/zero_shot.py",        label: "zero_shot.py",              module: "13" },
];

const LANGUAGE_META = [
  { file: "lang-c.md",      label: "C",          tag: "C",   badge: "lang-c",   desc: "Systems · Memory · Pointers",    icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M16.5921 9.1962s-.354-3.298-3.627-3.39c-3.2741-.09-4.9552 2.474-4.9552 6.14 0 3.6651 1.858 6.5972 5.0451 6.5972 3.184 0 3.5381-3.665 3.5381-3.665l6.1041.365s.36 3.31-2.196 5.836c-2.552 2.5241-5.6901 2.9371-7.8762 2.9201-2.19-.017-5.2261.034-8.1602-2.97-2.938-3.0101-3.436-5.9302-3.436-8.8002 0-2.8701.556-6.6702 4.047-9.5502C7.444.72 9.849 0 12.254 0c10.0422 0 10.7172 9.2602 10.7172 9.2602z"/></svg>` },
  { file: "lang-cpp.md",    label: "C++",         tag: "C++", badge: "lang-cpp", desc: "OOP · RAII · Templates · STL",   icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M22.394 6c-.167-.29-.398-.543-.652-.69L12.926.22c-.509-.294-1.34-.294-1.848 0L2.26 5.31c-.508.293-.923 1.013-.923 1.6v10.18c0 .294.104.62.271.91.167.29.398.543.652.69l8.816 5.09c.508.293 1.34.293 1.848 0l8.816-5.09c.254-.147.485-.4.652-.69.167-.29.27-.616.27-.91V6.91c.003-.294-.1-.62-.268-.91zM12 19.11c-3.92 0-7.109-3.19-7.109-7.11 0-3.92 3.19-7.11 7.11-7.11a7.133 7.133 0 016.156 3.553l-3.076 1.78a3.567 3.567 0 00-3.08-1.78A3.56 3.56 0 008.444 12 3.56 3.56 0 0012 15.555a3.57 3.57 0 003.08-1.778l3.078 1.78A7.135 7.135 0 0112 19.11zm7.11-6.715h-.79v.79h-.79v-.79h-.79v-.79h.79v-.79h.79v.79h.79zm2.962 0h-.79v.79h-.79v-.79h-.79v-.79h.79v-.79h.79v.79h.79z"/></svg>` },
  { file: "lang-python.md", label: "Python",      tag: "PY",  badge: "lang-py",  desc: "Internals · Async · Decorators", icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M14.25.18l.9.2.73.26.59.3.45.32.34.34.25.34.16.33.1.3.04.26.02.2-.01.13V8.5l-.05.63-.13.55-.21.46-.26.38-.3.31-.33.25-.35.19-.35.14-.33.1-.3.07-.26.04-.21.02H8.77l-.69.05-.59.14-.5.22-.41.27-.33.32-.27.35-.2.36-.15.37-.1.35-.07.32-.04.27-.02.21v3.06H3.17l-.21-.03-.28-.07-.32-.12-.35-.18-.36-.26-.36-.36-.35-.46-.32-.59-.28-.73-.21-.88-.14-1.05-.05-1.23.06-1.22.16-1.04.24-.87.32-.71.36-.57.4-.44.42-.33.42-.24.4-.16.36-.1.32-.05.24-.01h.16l.06.01h8.16v-.83H6.18l-.01-2.75-.02-.37.05-.34.11-.31.17-.28.25-.26.31-.23.38-.2.44-.18.51-.15.58-.12.64-.1.71-.06.77-.04.84-.02 1.27.05zm-6.3 1.98l-.23.33-.08.41.08.41.23.34.33.22.41.09.41-.09.33-.22.23-.34.08-.41-.08-.41-.23-.33-.33-.22-.41-.09-.41.09zm13.09 3.95l.28.06.32.12.35.18.36.27.36.35.35.47.32.59.28.73.21.88.14 1.04.05 1.23-.06 1.23-.16 1.04-.24.86-.32.71-.36.57-.4.45-.42.33-.42.24-.4.16-.36.09-.32.05-.24.02-.16-.01h-8.22v.82h5.84l.01 2.76.02.36-.05.34-.11.31-.17.29-.25.25-.31.24-.38.2-.44.17-.51.15-.58.13-.64.09-.71.07-.77.04-.84.01-1.27-.04-1.07-.14-.9-.2-.73-.25-.59-.3-.45-.33-.34-.34-.25-.34-.16-.33-.1-.3-.04-.25-.02-.2.01-.13v-5.34l.05-.64.13-.54.21-.46.26-.38.3-.32.33-.24.35-.2.35-.14.33-.1.3-.06.26-.04.21-.02.13-.01h5.84l.69-.05.59-.14.5-.21.41-.28.33-.32.27-.35.2-.36.15-.36.1-.35.07-.32.04-.28.02-.21V6.07h2.09l.14.01zm-6.47 14.25l-.23.33-.08.41.08.41.23.33.33.23.41.08.41-.08.33-.23.23-.33.08-.41-.08-.41-.23-.33-.33-.23-.41-.08-.41.08z"/></svg>` },
  { file: "lang-js.md",     label: "JavaScript",  tag: "JS",  badge: "lang-js",  desc: "V8 · Event Loop · Promises",     icon: `<svg role="img" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M0 0h24v24H0V0zm22.034 18.276c-.175-1.095-.888-2.015-3.003-2.873-.736-.345-1.554-.585-1.797-1.14-.091-.33-.105-.51-.046-.705.15-.646.915-.84 1.515-.66.39.12.75.42.976.9 1.034-.676 1.034-.676 1.755-1.125-.27-.42-.404-.601-.586-.78-.63-.705-1.469-1.065-2.834-1.034l-.705.089c-.676.165-1.32.525-1.71 1.005-1.14 1.291-.811 3.541.569 4.471 1.365 1.02 3.361 1.244 3.616 2.205.24 1.17-.87 1.545-1.966 1.41-.811-.18-1.26-.586-1.755-1.336l-1.83 1.051c.21.48.45.689.81 1.109 1.74 1.756 6.09 1.666 6.871-1.004.029-.09.24-.705.074-1.65l.046.067zm-8.983-7.245h-2.248c0 1.938-.009 3.864-.009 5.805 0 1.232.063 2.363-.138 2.711-.33.689-1.18.601-1.566.48-.396-.196-.597-.466-.83-.855-.063-.105-.11-.196-.127-.196l-1.825 1.125c.305.63.75 1.172 1.324 1.517.855.51 2.004.675 3.207.405.783-.226 1.458-.691 1.811-1.411.51-.93.402-2.07.397-3.346.012-2.054 0-4.109 0-6.179l.004-.056z"/></svg>` },
];

const PROJECT_META = [
  // Module 01 — Math for ML
  { file: "m01a.md", label: "PCA Image Compressor",        module: "01", difficulty: "Beginner",     type: "guided"   },
  { file: "m01b.md", label: "Gradient Descent Animator",   module: "01", difficulty: "Beginner",     type: "thinking" },
  { file: "m01c.md", label: "Bayesian Spam Classifier",    module: "01", difficulty: "Intermediate", type: "thinking" },
  // Module 02 — ML Basics
  { file: "m02a.md", label: "Titanic ML Pipeline",         module: "02", difficulty: "Beginner",     type: "guided"   },
  { file: "m02b.md", label: "Credit Risk Predictor",       module: "02", difficulty: "Intermediate", type: "thinking" },
  { file: "m02c.md", label: "Customer Churn XGBoost",      module: "02", difficulty: "Intermediate", type: "thinking" },
  // Module 03 — Databases & Vector DBs
  { file: "m03a.md", label: "Semantic Code Search",        module: "03", difficulty: "Intermediate", type: "guided"   },
  { file: "m03b.md", label: "Hybrid BM25+Dense Search",    module: "03", difficulty: "Intermediate", type: "thinking" },
  { file: "m03c.md", label: "Recipe Recommender",          module: "03", difficulty: "Intermediate", type: "thinking" },
  // Module 04 — Backend with Flask
  { file: "m04a.md", label: "Production ML API",           module: "04", difficulty: "Intermediate", type: "guided"   },
  { file: "m04b.md", label: "Rate-Limited API Gateway",    module: "04", difficulty: "Intermediate", type: "thinking" },
  { file: "m04c.md", label: "Real-Time WebSocket Server",  module: "04", difficulty: "Advanced",     type: "thinking" },
  // Module 05 — Deep Learning & MLOps
  { file: "m05a.md", label: "NN Training Dashboard",       module: "05", difficulty: "Intermediate", type: "guided"   },
  { file: "m05b.md", label: "ONNX Export Pipeline",        module: "05", difficulty: "Advanced",     type: "thinking" },
  { file: "m05c.md", label: "Drift Detection System",      module: "05", difficulty: "Advanced",     type: "thinking" },
  // Module 06 — GenAI Core
  { file: "m06a.md", label: "Word Analogy Explorer",       module: "06", difficulty: "Intermediate", type: "guided"   },
  { file: "m06b.md", label: "Semantic Document Retrieval", module: "06", difficulty: "Intermediate", type: "thinking" },
  { file: "m06c.md", label: "Embedding Eval Benchmark",    module: "06", difficulty: "Advanced",     type: "thinking" },
  // Module 07 — Transformers
  { file: "m07a.md", label: "Shakespeare GPT",             module: "07", difficulty: "Advanced",     type: "guided"   },
  { file: "m07b.md", label: "Custom BPE Tokenizer",        module: "07", difficulty: "Advanced",     type: "thinking" },
  { file: "m07c.md", label: "Mini Translation Model",      module: "07", difficulty: "Expert",       type: "thinking" },
  // Module 08 — RAG
  { file: "m08a.md", label: "Personal Document Q&A",       module: "08", difficulty: "Advanced",     type: "guided"   },
  { file: "m08b.md", label: "Multi-Source Research Bot",   module: "08", difficulty: "Advanced",     type: "thinking" },
  { file: "m08c.md", label: "RAGAS-Driven RAG Optimizer",  module: "08", difficulty: "Expert",       type: "thinking" },
  // Module 09 — Fine-Tuning
  { file: "m09a.md", label: "Domain LoRA Fine-Tuner",      module: "09", difficulty: "Expert",       type: "guided"   },
  { file: "m09b.md", label: "Instruction Dataset Builder", module: "09", difficulty: "Advanced",     type: "thinking" },
  { file: "m09c.md", label: "PEFT Method Comparison",      module: "09", difficulty: "Expert",       type: "thinking" },
  // Module 10 — Agents
  { file: "m10a.md", label: "Multi-Agent Crypto Analyst",  module: "10", difficulty: "Expert",       type: "guided"   },
  { file: "m10b.md", label: "Research Assistant Agent",    module: "10", difficulty: "Advanced",     type: "thinking" },
  { file: "m10c.md", label: "Self-Debugging Code Agent",   module: "10", difficulty: "Expert",       type: "thinking" },
  // Module 11 — Deployment
  { file: "m11a.md", label: "A/B Testing Deploy API",      module: "11", difficulty: "Advanced",     type: "guided"   },
  { file: "m11b.md", label: "Canary Deployment Monitor",   module: "11", difficulty: "Advanced",     type: "thinking" },
  { file: "m11c.md", label: "Multi-Model Serving Gateway", module: "11", difficulty: "Expert",       type: "thinking" },
  // Module 12 — RLHF & Alignment
  { file: "m12a.md", label: "Mini-RLHF Preference Tuner", module: "12", difficulty: "Expert",       type: "guided"   },
  { file: "m12b.md", label: "Reward Model Evaluator",      module: "12", difficulty: "Expert",       type: "thinking" },
  { file: "m12c.md", label: "DPO Dataset Curator",         module: "12", difficulty: "Expert",       type: "thinking" },
  // Module 13 — Multimodal
  { file: "m13a.md", label: "Image-Text Hybrid Search",    module: "13", difficulty: "Expert",       type: "guided"   },
  { file: "m13b.md", label: "Zero-Shot Image Classifier",  module: "13", difficulty: "Advanced",     type: "thinking" },
  { file: "m13c.md", label: "Visual Q&A System",           module: "13", difficulty: "Expert",       type: "thinking" },
  // Module 14 — Frontend
  { file: "m14a.md", label: "Full-Stack AI Blog",          module: "14", difficulty: "Advanced",     type: "guided"   },
  { file: "m14b.md", label: "Real-Time Voice Assistant",   module: "14", difficulty: "Expert",       type: "thinking" },
  { file: "m14c.md", label: "Interactive ML Dashboard",    module: "14", difficulty: "Advanced",     type: "thinking" },
];

// ── State ───────────────────────────────────────────────────────
let state = {
  currentFile:      null,
  currentType:      null,
  progress:         loadProgress(),
  projectsProgress: loadProjectsProgress(),
  focusMode:        false,
  floatingMode:     false,
  lightMode:        localStorage.getItem("aiml_theme") ? localStorage.getItem("aiml_theme") === "light" : true,
  notes:            localStorage.getItem("aiml_notes") || "",
  activeTabIdx:     0,
};

// ── Persistence ─────────────────────────────────────────────────
function loadProgress() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; }
  catch { return {}; }
}

function loadProjectsProgress() {
  try { return JSON.parse(localStorage.getItem(LS_KEY_PROJECTS)) || {}; }
  catch { return {}; }
}

function saveProgress()         { localStorage.setItem(LS_KEY,          JSON.stringify(state.progress));         }
function saveProjectsProgress() { localStorage.setItem(LS_KEY_PROJECTS, JSON.stringify(state.projectsProgress)); }

// ── Loading screen ───────────────────────────────────────────────
function hideLoadingScreen() {
  const elapsed = Date.now() - LOAD_START;
  const delay   = Math.max(0, MIN_LOAD_MS - elapsed);
  setTimeout(() => {
    const screen = document.getElementById("loading-screen");
    if (screen) screen.classList.add("fade-out");
  }, delay);
}

// ── Greeting ─────────────────────────────────────────────────────
function getGreetingPhrase() {
  const h = new Date().getHours();
  if (h < 5)  return "Working late, Harshit";
  if (h < 12) return "Good morning, Harshit";
  if (h < 17) return "Good afternoon, Harshit";
  if (h < 21) return "Good evening, Harshit";
  return "Burning midnight oil, Harshit";
}

function updateGreeting() {
  const done  = MODULE_META.filter(m => state.progress[m.file] === "done").length;
  const pdone = PROJECT_META.filter(p => state.projectsProgress[p.file] === "done").length;
  const pct   = Math.round((done / MODULE_META.length) * 100);

  const timeEl  = document.getElementById("greeting-time");
  const statsEl = document.getElementById("greeting-stats");

  if (timeEl)  timeEl.textContent  = getGreetingPhrase();
  if (statsEl) statsEl.textContent = `${done}/${MODULE_META.length} modules · ${pdone}/${PROJECT_META.length} projects · ${pct}% complete`;

  updateRing(pct);
}

// ── Progress Ring ─────────────────────────────────────────────────
function updateRing(pct) {
  const circumference = 2 * Math.PI * 50; // 314.16
  const offset = circumference * (1 - pct / 100);
  const ringFill = document.getElementById("ring-fill");
  const ringPct  = document.getElementById("ring-pct");
  if (ringFill) ringFill.style.strokeDashoffset = offset;
  if (ringPct)  ringPct.textContent = pct + "%";
}

// ── marked.js config ────────────────────────────────────────────
marked.setOptions({
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
    return hljs.highlightAuto(code).value;
  },
  breaks: false,
  gfm: true,
});

// ── DOM helper ───────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ── Theme Toggle ─────────────────────────────────────────────────
function applyTheme(light) {
  document.body.classList.toggle("light-mode", light);

  const hljsLink = document.getElementById("hljs-theme");
  if (hljsLink) {
    hljsLink.href = light
      ? "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css"
      : "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/base16/onedark.min.css";
  }

  const iconWrap = document.getElementById("dock-theme-icon");
  if (iconWrap) {
    iconWrap.innerHTML = light
      // sun icon
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
           <circle cx="12" cy="12" r="5"/>
           <line x1="12" y1="1" x2="12" y2="3"/>
           <line x1="12" y1="21" x2="12" y2="23"/>
           <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
           <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
           <line x1="1" y1="12" x2="3" y2="12"/>
           <line x1="21" y1="12" x2="23" y2="12"/>
           <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
           <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
         </svg>`
      // moon icon
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
           <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
         </svg>`;
  }

  const dockTheme = $("dock-theme");
  if (dockTheme) {
    dockTheme.classList.toggle("active-dock", light);
    const lbl = dockTheme.querySelector(".dock-label");
    if (lbl) lbl.textContent = light ? "Dark Mode" : "Light Mode";
  }
}

function toggleTheme() {
  state.lightMode = !state.lightMode;
  localStorage.setItem("aiml_theme", state.lightMode ? "light" : "dark");
  applyTheme(state.lightMode);
}


// ── Floating Sidebar Mode ───────────────────────────────────────
function toggleFloatingMode() {
  state.floatingMode = !state.floatingMode;
  document.body.classList.toggle("floating-mode", state.floatingMode);
}

document.addEventListener("DOMContentLoaded", () => {
  const pinBtn = document.getElementById("btn-pin-sidebar");
  if (pinBtn) {
    pinBtn.addEventListener("click", toggleFloatingMode);
  }
});

// ── Focus Mode ───────────────────────────────────────────────────
function toggleFocusMode() {
  state.focusMode = !state.focusMode;
  document.body.classList.toggle("focus-mode", state.focusMode);

  const dockFocus = $("dock-focus");
  const topFocus  = $("btn-focus");

  if (dockFocus) {
    dockFocus.classList.toggle("active-dock", state.focusMode);
    const lbl = dockFocus.querySelector(".dock-label");
    if (lbl) lbl.textContent = state.focusMode ? "Pin Sidebar" : "Focus Mode";
  }
  if (topFocus) topFocus.classList.toggle("active", state.focusMode);

  if (state.focusMode) {
    if (document.documentElement.requestFullscreen) {
      document.documentElement.requestFullscreen().catch(() => {});
    }
  } else {
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    }
    $("sidebar").classList.remove("focus-revealed");
  }
}

// Sync if user presses Escape to exit fullscreen
document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement && state.focusMode) {
    state.focusMode = false;
    document.body.classList.remove("focus-mode");
    $("sidebar").classList.remove("focus-revealed");
    const dockFocus = $("dock-focus");
    if (dockFocus) {
      dockFocus.classList.remove("active-dock");
      const lbl = dockFocus.querySelector(".dock-label");
      if (lbl) lbl.textContent = "Focus";
    }
    const topFocus = $("btn-focus");
    if (topFocus) topFocus.classList.remove("active");
  }
});

// ── Sidebar hover in focus mode ───────────────────────────────────
(function setupSidebarHover() {
  const hoverZone = $("sidebar-hover-zone");
  const sidebar   = $("sidebar");
  if (!hoverZone || !sidebar) return;

  hoverZone.addEventListener("mouseenter", () => {
    if (state.focusMode || state.floatingMode) sidebar.classList.add("focus-revealed");
  });

  sidebar.addEventListener("mouseleave", () => {
    if (state.focusMode || state.floatingMode) sidebar.classList.remove("focus-revealed");
  });
})();

// ── Mobile sidebar ───────────────────────────────────────────────
function openSidebar() {
  $("sidebar").classList.add("mobile-open");
  $("sidebar-overlay").classList.add("visible");
  const toggle = $("sidebar-toggle");
  toggle.classList.add("open");
  toggle.setAttribute("aria-expanded", "true");
}

function closeSidebar() {
  $("sidebar").classList.remove("mobile-open");
  $("sidebar-overlay").classList.remove("visible");
  const toggle = $("sidebar-toggle");
  toggle.classList.remove("open");
  toggle.setAttribute("aria-expanded", "false");
}

// ── Content fade helper ──────────────────────────────────────────
function fadeContent(fn) {
  const ca = $("content-area");
  ca.style.opacity = "0";
  requestAnimationFrame(() => {
    setTimeout(() => {
      fn();
      ca.style.opacity = "1";
    }, 160);
  });
}

// ── Build module sidebar nav ─────────────────────────────────────
function buildNav() {
  const ul = $("module-list");
  ul.innerHTML = "";

  MODULE_META.forEach((mod, idx) => {
    const completed = state.progress[mod.file] === "done";
    const li        = document.createElement("li");
    li.dataset.file = mod.file;
    li.dataset.idx  = idx;
    li.className    = completed ? "completed" : "";
    li.setAttribute("role", "listitem");
    li.setAttribute("tabindex", "0");
    li.setAttribute("aria-label", `Module ${mod.tag}: ${mod.label}${completed ? " (completed)" : ""}`);

    li.innerHTML = `
      <span class="module-num" aria-hidden="true">${mod.tag}</span>
      <span class="module-name">${mod.label}</span>
      <span class="status-dot" aria-hidden="true"></span>
    `;

    li.addEventListener("click", () => { openModule(mod.file, mod.label); closeSidebar(); });
    li.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openModule(mod.file, mod.label); closeSidebar(); }
    });
    ul.appendChild(li);
  });

  updateProgressBar();
}

// ── Build project sidebar nav ────────────────────────────────────
function buildProjectNav() {
  const ul = $("project-list");
  ul.innerHTML = "";

  PROJECT_META.forEach((proj, idx) => {
    const completed = state.projectsProgress[proj.file] === "done";
    const li        = document.createElement("li");
    li.dataset.file = proj.file;
    li.dataset.idx  = idx;
    li.className    = "project-item" + (completed ? " completed" : "");
    li.setAttribute("role", "listitem");
    li.setAttribute("tabindex", "0");
    li.setAttribute("aria-label", `Project: ${proj.label}${completed ? " (completed)" : ""}`);

    const diffClass = proj.difficulty ? `diff-${proj.difficulty.toLowerCase()}` : "";
    li.innerHTML = `
      <span class="module-num" aria-hidden="true">${proj.type === "guided" ? "▶" : "◈"} ${proj.module}</span>
      <span class="module-name">${proj.label}</span>
      <span class="proj-diff-dot ${diffClass}" aria-hidden="true" title="${proj.difficulty || ""}"></span>
      <span class="status-dot" aria-hidden="true"></span>
    `;

    li.addEventListener("click", () => { openProject(proj.file, proj.label); closeSidebar(); });
    li.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openProject(proj.file, proj.label); closeSidebar(); }
    });
    ul.appendChild(li);
  });
}

// ── Build code sidebar nav ────────────────────────────────────────
function buildCodeNav() {
  const ul = $("code-list");
  ul.innerHTML = "";

  let currentModule = "";

  CODE_META.forEach((code, idx) => {
    if (code.module !== currentModule) {
      currentModule = code.module;
      const modMeta = MODULE_META.find(m => m.tag === currentModule);
      const groupHeader = document.createElement("div");
      groupHeader.className = "module-group-header";
      groupHeader.textContent = `${currentModule} · ${modMeta ? modMeta.label : ""}`;
      ul.appendChild(groupHeader);
    }

    const li = document.createElement("li");
    li.dataset.file = code.file;
    li.dataset.idx  = idx;
    li.className    = "code-item";
    li.setAttribute("role", "listitem");
    li.setAttribute("tabindex", "0");
    li.setAttribute("aria-label", `Code file: ${code.label}`);

    li.innerHTML = `
      <span class="module-num" aria-hidden="true" style="opacity:0.4;">.py</span>
      <span class="module-name">${code.label}</span>
    `;

    li.addEventListener("click", () => { openCode(code.file, code.label); closeSidebar(); });
    li.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openCode(code.file, code.label); closeSidebar(); }
    });
    ul.appendChild(li);
  });
}

// ── Build language sidebar nav ────────────────────────────────────
function buildLanguageNav() {
  const ul = $("language-list");
  ul.innerHTML = "";

  LANGUAGE_META.forEach((lang, idx) => {
    const li = document.createElement("li");
    li.dataset.file = lang.file;
    li.dataset.idx  = idx;
    li.className    = "project-item";
    li.setAttribute("role", "listitem");
    li.setAttribute("tabindex", "0");
    li.setAttribute("aria-label", `Language: ${lang.label}`);

    li.innerHTML = `
      <span class="module-num lang-nav-icon" aria-hidden="true">${lang.icon}</span>
      <span class="module-name">${lang.label}</span>
    `;

    li.addEventListener("click", () => { openLanguage(lang.file, lang.label); closeSidebar(); });
    li.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openLanguage(lang.file, lang.label); closeSidebar(); }
    });
    ul.appendChild(li);
  });
}

// ── Build languages welcome grid ─────────────────────────────────
function buildLanguagesGrid() {
  const grid = $("languages-grid");
  if (!grid) return;
  grid.innerHTML = "";

  LANGUAGE_META.forEach((lang, idx) => {
    const card = document.createElement("div");
    card.className = `welcome-module-card language-card wlc-${idx}`;
    card.style.animationDelay = `${idx * 35}ms`;
    card.setAttribute("role", "listitem");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `Open language: ${lang.label}`);

    card.innerHTML = `
      <div class="wmc-icon" aria-hidden="true">${lang.icon}</div>
      <span class="wmc-num" aria-hidden="true">LANGUAGE</span>
      <div class="wmc-title">${lang.label}</div>
      <div class="wmc-lang-desc">${lang.desc}</div>
      <span class="lang-badge ${lang.badge}" aria-label="${lang.label}">${lang.tag}</span>
      <div class="wmc-status" aria-hidden="true"></div>
    `;

    card.addEventListener("click", () => openLanguage(lang.file, lang.label));
    card.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openLanguage(lang.file, lang.label); }
    });
    grid.appendChild(card);
  });
}

// ── Build module welcome grid ────────────────────────────────────
function buildWelcomeGrid() {
  const grid = $("welcome-grid");
  if (!grid) return;
  grid.innerHTML = "";

  MODULE_META.forEach((mod, idx) => {
    const completed = state.progress[mod.file] === "done";
    const card = document.createElement("div");
    card.className = `welcome-module-card wmc-${idx}`;
    card.style.animationDelay = `${idx * 30}ms`;
    card.setAttribute("role", "listitem");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `Open module ${mod.tag}: ${mod.label}`);

    card.innerHTML = `
      <div class="wmc-icon" aria-hidden="true">${mod.icon}</div>
      <span class="wmc-num" aria-hidden="true">MODULE ${mod.tag}</span>
      <div class="wmc-title">${mod.label}</div>
      <div class="wmc-status ${completed ? "done" : ""}" aria-hidden="true"></div>
    `;

    card.addEventListener("click", () => openModule(mod.file, mod.label));
    card.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openModule(mod.file, mod.label); }
    });
    grid.appendChild(card);
  });
}

// ── Build projects welcome grid ──────────────────────────────────
function buildProjectsGrid() {
  const grid = $("projects-grid");
  if (!grid) return;
  grid.innerHTML = "";

  PROJECT_META.forEach((proj, idx) => {
    const completed = state.projectsProgress[proj.file] === "done";
    const card = document.createElement("div");
    card.className = `welcome-module-card project-card wpc-${idx}`;
    card.style.animationDelay = `${idx * 28}ms`;
    card.setAttribute("role", "listitem");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `Open project: ${proj.label} (${proj.difficulty})`);

    const projIcon = proj.type === "guided"
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
    card.innerHTML = `
      <div class="wmc-icon" aria-hidden="true">${projIcon}</div>
      <span class="wmc-num" aria-hidden="true">MOD ${proj.module} · <span class="proj-type-tag type-${proj.type}">${proj.type === "guided" ? "GUIDED" : "THINKING"}</span></span>
      <div class="wmc-title">${proj.label}</div>
      <span class="difficulty-badge diff-${proj.difficulty.toLowerCase()}" aria-label="Difficulty: ${proj.difficulty}">${proj.difficulty}</span>
      <div class="wmc-status ${completed ? "done" : ""}" aria-hidden="true"></div>
    `;

    card.addEventListener("click", () => openProject(proj.file, proj.label));
    card.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openProject(proj.file, proj.label); }
    });
    grid.appendChild(card);
  });
}

// ── Build code welcome — grouped by module ───────────────────────
function buildCodeGrid() {
  const container = $("code-by-module");
  if (!container) return;
  container.innerHTML = "";

  // Group files by module tag
  const groups = {};
  CODE_META.forEach(code => {
    if (!groups[code.module]) groups[code.module] = [];
    groups[code.module].push(code);
  });

  let sectionIdx = 0;
  Object.keys(groups).sort().forEach(moduleTag => {
    const modMeta = MODULE_META.find(m => m.tag === moduleTag);
    const section = document.createElement("div");
    section.className = "code-module-section";

    const header = document.createElement("div");
    header.className = "code-module-header";
    header.innerHTML = `
      <span class="code-mod-tag">Module ${moduleTag}</span>
      <span class="code-mod-name">${modMeta ? modMeta.label : ""}</span>
    `;
    section.appendChild(header);

    const grid = document.createElement("div");
    grid.className = "code-module-grid";

    groups[moduleTag].forEach((code, fileIdx) => {
      const ext  = code.file.split(".").pop();
      const name = code.file.split("/").pop();

      const card = document.createElement("div");
      card.className = "code-file-card";
      card.style.animationDelay = `${(sectionIdx * 4 + fileIdx) * 25}ms`;
      card.setAttribute("role", "listitem");
      card.setAttribute("tabindex", "0");
      card.setAttribute("aria-label", `Open code file: ${name}`);

      card.innerHTML = `
        <span class="code-file-ext">${ext}</span>
        <div class="code-file-name">${name}</div>
      `;

      card.addEventListener("click", () => openCode(code.file, code.label));
      card.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openCode(code.file, code.label); }
      });
      grid.appendChild(card);
    });

    section.appendChild(grid);
    container.appendChild(section);
    sectionIdx++;
  });
}

// ── Progress bar ────────────────────────────────────────────────
function updateProgressBar() {
  const total = MODULE_META.length;
  const done  = MODULE_META.filter(m => state.progress[m.file] === "done").length;
  const pct   = Math.round((done / total) * 100);

  $("progress-count").textContent    = done;
  $("module-total").textContent      = total;
  $("progress-bar-fill").style.width = `${(done / total) * 100}%`;

  const container = $("progress-container");
  if (container) container.setAttribute("aria-valuenow", done);

  updateRing(pct);
  updateGreeting();
}

// ── Open module ──────────────────────────────────────────────────
async function openModule(file, label) {
  state.currentFile = file;
  state.currentType = "module";
  recordStudyActivity();

  document.querySelectorAll("#module-list li").forEach(li =>
    li.classList.toggle("active", li.dataset.file === file));
  document.querySelectorAll("#project-list li").forEach(li =>
    li.classList.remove("active"));
  document.querySelectorAll("#code-list li").forEach(li =>
    li.classList.remove("active"));

  $("breadcrumb").textContent = label;

  const isDone = state.progress[file] === "done";
  const btn    = $("btn-complete");
  btn.classList.remove("hidden");
  btn.textContent = isDone ? "✓ COMPLETED" : "MARK COMPLETE";
  btn.className   = "complete-btn" + (isDone ? " done" : "");

  fadeContent(async () => {
    $("welcome-screen").classList.add("hidden");
    $("doc-view").classList.remove("hidden");
    $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#444;font-size:13px;">Loading ${file}…</p>`;

    try {
      const res = await fetch(`docs/${file}`);
      if (!res.ok) {
        if (res.status === 404) { renderNotReady(file, label); return; }
        throw new Error(`HTTP ${res.status}`);
      }
      renderMarkdown(await res.text());
    } catch (err) {
      $("doc-rendered").innerHTML = `<div class="error-box"><strong>Error loading ${file}</strong><br>${err.message}</div>`;
    }
  });
}

// ── Open project ─────────────────────────────────────────────────
async function openProject(file, label) {
  state.currentFile = file;
  state.currentType = "project";
  recordStudyActivity();

  document.querySelectorAll("#module-list li").forEach(li =>
    li.classList.remove("active"));
  document.querySelectorAll("#project-list li").forEach(li =>
    li.classList.toggle("active", li.dataset.file === file));
  document.querySelectorAll("#code-list li").forEach(li =>
    li.classList.remove("active"));

  $("breadcrumb").textContent = "PROJECT — " + label;

  const isDone = state.projectsProgress[file] === "done";
  const btn    = $("btn-complete");
  btn.classList.remove("hidden");
  btn.textContent = isDone ? "✓ COMPLETED" : "MARK COMPLETE";
  btn.className   = "complete-btn" + (isDone ? " done" : "");

  fadeContent(async () => {
    $("welcome-screen").classList.add("hidden");
    $("doc-view").classList.remove("hidden");
    $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#444;font-size:13px;">Loading ${file}…</p>`;

    try {
      const res = await fetch(`docs/projects/${file}`);
      if (!res.ok) {
        if (res.status === 404) { renderNotReady(file, label); return; }
        throw new Error(`HTTP ${res.status}`);
      }
      renderMarkdown(await res.text());
    } catch (err) {
      $("doc-rendered").innerHTML = `<div class="error-box"><strong>Error loading ${file}</strong><br>${err.message}</div>`;
    }
  });
}

// ── Open code file ───────────────────────────────────────────────
async function openCode(file, label) {
  state.currentFile = file;
  state.currentType = "code";

  document.querySelectorAll("#module-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#project-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#code-list li").forEach(li =>
    li.classList.toggle("active", li.dataset.file === file));

  $("breadcrumb").textContent = "CODE — " + label;
  $("btn-complete").classList.add("hidden");

  fadeContent(async () => {
    $("welcome-screen").classList.add("hidden");
    $("doc-view").classList.remove("hidden");
    $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#444;font-size:13px;">Loading ${file}…</p>`;

    try {
      const res = await fetch(`src/${file}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      let ext = file.split(".").pop();
      if (ext === "py")             ext = "python";
      else if (ext === "cpp" || ext === "h") ext = "cpp";
      else if (ext === "sql")       ext = "sql";

      renderMarkdown("```" + ext + "\n" + text + "\n```");
    } catch (err) {
      $("doc-rendered").innerHTML = `<div class="error-box"><strong>Error loading ${file}</strong><br>${err.message}</div>`;
    }
  });
}

// ── Open language file ───────────────────────────────────────────
async function openLanguage(file, label) {
  state.currentFile = file;
  state.currentType = "language";
  recordStudyActivity();

  document.querySelectorAll("#module-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#project-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#code-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#language-list li").forEach(li =>
    li.classList.toggle("active", li.dataset.file === file));

  $("breadcrumb").textContent = "LANGUAGE — " + label;
  $("btn-complete").classList.add("hidden");

  fadeContent(async () => {
    $("welcome-screen").classList.add("hidden");
    $("doc-view").classList.remove("hidden");
    $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#444;font-size:13px;">Loading ${file}…</p>`;

    try {
      const res = await fetch(`docs/${file}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      renderMarkdown(await res.text());
    } catch (err) {
      $("doc-rendered").innerHTML = `<div class="error-box"><strong>Error loading ${file}</strong><br>${err.message}</div>`;
    }
  });
}

// ── Render markdown ──────────────────────────────────────────────
function renderMarkdown(md) {
  $("doc-rendered").innerHTML = marked.parse(md);

  $("doc-rendered").querySelectorAll("pre code").forEach(block => hljs.highlightElement(block));

  $("doc-rendered").querySelectorAll("a[href]").forEach(a => {
    const basename = a.getAttribute("href").split("/").pop();
    const mod  = MODULE_META.find(m => m.file === basename);
    const proj = PROJECT_META.find(p => p.file === basename);
    if (mod)       a.addEventListener("click", e => { e.preventDefault(); openModule(mod.file,   mod.label);   });
    else if (proj) a.addEventListener("click", e => { e.preventDefault(); openProject(proj.file, proj.label); });
  });

  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetPromise([$("doc-rendered")]).catch(console.error);
  }

  $("content-area").scrollTop = 0;
}

// ── Not-ready placeholder ────────────────────────────────────────
function renderNotReady(file, label) {
  const moduleNum = file.split("-")[0];
  $("doc-rendered").innerHTML = `
    <div style="border:1px solid rgba(255,255,255,0.1);padding:40px;background:#0d0d0d;max-width:600px;border-radius:2px;">
      <div style="display:inline-block;background:rgba(230,255,0,0.08);color:#E6FF00;font-family:monospace;
                  font-size:10px;letter-spacing:2px;padding:4px 12px;margin-bottom:20px;border:1px solid rgba(230,255,0,0.2);">
        NOT YET GENERATED
      </div>
      <h2 style="font-size:26px;font-weight:900;margin-bottom:12px;font-family:'Space Grotesk',sans-serif;">${label}</h2>
      <p style="font-family:monospace;color:#555;margin-bottom:20px;font-size:13px;">This content has not been generated yet.</p>
      <div style="border:1px solid rgba(255,255,255,0.08);padding:16px;font-family:monospace;font-size:12px;background:rgba(255,255,255,0.02);color:#666;">
        File: docs/${file}<br>
        Code: src/${moduleNum}-*/
      </div>
    </div>
  `;
}

// ── Toggle complete ──────────────────────────────────────────────
function toggleComplete() {
  if (!state.currentFile) return;

  const isProject   = state.currentType === "project";
  const progressObj = isProject ? state.projectsProgress : state.progress;
  const isDone      = progressObj[state.currentFile] === "done";

  if (isDone) delete progressObj[state.currentFile];
  else        progressObj[state.currentFile] = "done";

  if (isProject) {
    saveProjectsProgress();
    buildProjectNav();
    buildProjectsGrid();
  } else {
    saveProgress();
    buildNav();
    buildWelcomeGrid();
    updateProgressBar();
  }

  const nowDone = progressObj[state.currentFile] === "done";
  const btn = $("btn-complete");
  btn.textContent = nowDone ? "✓ COMPLETED" : "MARK COMPLETE";
  btn.className   = "complete-btn" + (nowDone ? " done" : "");

  const listId = isProject ? "project-list" : "module-list";
  document.querySelectorAll(`#${listId} li`).forEach(li => {
    if (li.dataset.file === state.currentFile) li.classList.toggle("completed", nowDone);
  });

  updateGreeting();
}

// ── Reset all progress ───────────────────────────────────────────
function resetProgress() {
  if (!confirm("Reset all progress? This cannot be undone.")) return;
  state.progress         = {};
  state.projectsProgress = {};
  saveProgress();
  saveProjectsProgress();
  buildNav();
  buildProjectNav();
  buildLanguageNav();
  buildCodeNav();
  buildWelcomeGrid();
  buildProjectsGrid();
  buildLanguagesGrid();
  buildCodeGrid();
  updateProgressBar();
}

// ── Go Home (roadmap) ────────────────────────────────────────────
function goHome() {
  state.currentFile = null;
  state.currentType = null;
  document.querySelectorAll("#module-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#project-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#language-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#code-list li").forEach(li => li.classList.remove("active"));

  fadeContent(() => {
    $("breadcrumb").textContent = "Roadmap";
    $("doc-view").classList.add("hidden");
    $("welcome-screen").classList.remove("hidden");
    $("btn-complete").classList.add("hidden");
    buildWelcomeGrid();
    buildProjectsGrid();
    buildLanguagesGrid();
    buildCodeGrid();
    updateGreeting();
  });
}

// ── Sidebar roadmap btn ──────────────────────────────────────────
$("btn-roadmap").addEventListener("click", goHome);

// ── Mobile sidebar events ────────────────────────────────────────
$("sidebar-toggle").addEventListener("click", () => {
  const isOpen = $("sidebar").classList.contains("mobile-open");
  isOpen ? closeSidebar() : openSidebar();
});

$("sidebar-overlay").addEventListener("click", closeSidebar);

// ── Focus button (topbar) ────────────────────────────────────────
$("btn-focus").addEventListener("click", toggleFocusMode);

// ── Keyboard shortcuts ───────────────────────────────────────────
document.addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    if (state.currentFile) toggleComplete();
    return;
  }
  if (e.key === "f" || e.key === "F") {
    // Only toggle focus if not typing in an input
    const tag = document.activeElement.tagName;
    if (tag !== "INPUT" && tag !== "TEXTAREA") toggleFocusMode();
    return;
  }
  if (e.key === "Escape") {
    if (state.focusMode) { toggleFocusMode(); return; }
    if ($("sidebar").classList.contains("mobile-open")) { closeSidebar(); return; }
    const searchModal = $("search-modal");
    if (searchModal && !searchModal.classList.contains("hidden")) return;
    goHome();
  }
});

// ── Sidebar Tabs — Dynamic Drum Selector ─────────────────────────
const ALL_NAV_LISTS = ["module-list", "project-list", "language-list", "code-list"];

function setActiveTab(activeIdx) {
  state.activeTabIdx = activeIdx;
  const tabs = Array.from(document.querySelectorAll(".sidebar-tab"));

  tabs.forEach((tab, i) => {
    const dist = Math.abs(i - activeIdx);
    tab.style.setProperty("--dist", dist);
    tab.classList.toggle("active", i === activeIdx);
    tab.setAttribute("aria-selected", i === activeIdx ? "true" : "false");
  });

  const target = tabs[activeIdx]?.getAttribute("data-target");
  ALL_NAV_LISTS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("hidden", id !== target);
  });
}

document.querySelectorAll(".sidebar-tab").forEach((tab, i) => {
  tab.addEventListener("click", () => setActiveTab(i));

  tab.addEventListener("mouseenter", () => {
    if (i === state.activeTabIdx) return;
    const tabs = Array.from(document.querySelectorAll(".sidebar-tab"));
    tabs.forEach((t, j) => {
      const distFromHover = Math.abs(j - i);
      const distFromActive = Math.abs(j - state.activeTabIdx);
      const blendedDist = Math.min(distFromHover * 0.5, distFromActive);
      t.style.setProperty("--dist", blendedDist.toFixed(2));
    });
  });

  tab.addEventListener("mouseleave", () => {
    setActiveTab(state.activeTabIdx);
  });
});


// ── Welcome Screen Tabs ──────────────────────────────────────────
document.querySelectorAll(".welcome-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".welcome-tab").forEach(t => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");

    const panelId = tab.getAttribute("data-panel");
    document.querySelectorAll(".welcome-panel").forEach(panel => {
      panel.classList.toggle("hidden", panel.id !== panelId);
    });

    // Rebuild panels on switch to keep animations fresh
    if (panelId === "modules-panel")   buildWelcomeGrid();
    if (panelId === "projects-panel")  buildProjectsGrid();
    if (panelId === "languages-panel") buildLanguagesGrid();
    if (panelId === "code-panel")      buildCodeGrid();
  });
});

// ── Dock button handlers ─────────────────────────────────────────
$("dock-home").addEventListener("click",   goHome);
$("dock-focus").addEventListener("click",  toggleFocusMode);
$("dock-theme").addEventListener("click",  toggleTheme);
$("dock-reset").addEventListener("click",  resetProgress);
$("dock-streak").addEventListener("click", goHome);

// ── Notes ────────────────────────────────────────────────────────
(function setupNotes() {
  const notesText = $("notes-content");
  if (notesText) {
    notesText.value = state.notes;
    notesText.addEventListener("input", e => {
      state.notes = e.target.value;
      localStorage.setItem("aiml_notes", state.notes);
    });
  }

  function toggleNotes() {
    const panel = $("notes-panel");
    if (panel) panel.classList.toggle("hidden");
  }

  $("dock-notes").addEventListener("click", toggleNotes);
  $("close-notes").addEventListener("click", () => $("notes-panel").classList.add("hidden"));
})();

// ── Search ───────────────────────────────────────────────────────
(function setupSearch() {
  const searchModal   = $("search-modal");
  const searchInput   = $("search-input");
  const searchResults = $("search-results");
  let selectedIndex   = -1;

  function openSearch() {
    searchModal.classList.remove("hidden");
    searchInput.value = "";
    searchResults.innerHTML = "";
    selectedIndex = -1;
    setTimeout(() => searchInput.focus(), 80);
  }

  function closeSearch() {
    searchModal.classList.add("hidden");
  }

  $("dock-search").addEventListener("click", openSearch);

  document.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "k") {
      e.preventDefault();
      openSearch();
    }
    if (e.key === "Escape" && !searchModal.classList.contains("hidden")) {
      e.stopPropagation();
      closeSearch();
    }
  });

  searchInput.addEventListener("input", e => {
    const q = e.target.value.trim();
    searchResults.innerHTML = "";
    selectedIndex = -1;
    if (!q) return;

    const results = fuse.search(q).slice(0, 12);

    results.forEach((result, idx) => {
      const item = result.item;
      const li = document.createElement("li");
      li.innerHTML = `<span>${item.label}</span><span class="sr-type">${item.type}</span>`;
      li.addEventListener("click", () => { closeSearch(); item.execute(); });
      li.addEventListener("mouseover", () => {
        Array.from(searchResults.children).forEach(c => c.classList.remove("selected"));
        li.classList.add("selected");
        selectedIndex = idx;
      });
      searchResults.appendChild(li);
    });
  });

  searchInput.addEventListener("keydown", e => {
    const items = Array.from(searchResults.children);
    if (e.key === "ArrowDown") {
      e.preventDefault();
      selectedIndex = (selectedIndex + 1) % items.length;
      updateSel();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selectedIndex = (selectedIndex - 1 + items.length) % items.length;
      updateSel();
    } else if (e.key === "Enter" && selectedIndex >= 0) {
      e.preventDefault();
      items[selectedIndex]?.click();
    }

    function updateSel() {
      items.forEach(c => c.classList.remove("selected"));
      if (items[selectedIndex]) items[selectedIndex].classList.add("selected");
    }
  });

  searchModal.addEventListener("click", e => {
    if (e.target === searchModal) closeSearch();
  });
})();

// ── Study Streak & Heatmap ───────────────────────────────────────
function recordStudyActivity() {
  const today = new Date().toISOString().split('T')[0];
  const log = JSON.parse(localStorage.getItem('aiml_study_log') || '{}');
  log[today] = (log[today] || 0) + 1;
  localStorage.setItem('aiml_study_log', JSON.stringify(log));
  updateStreak();
}

function computeStreak() {
  const log = JSON.parse(localStorage.getItem('aiml_study_log') || '{}');
  let streak = 0;
  const d = new Date();
  while (true) {
    const key = d.toISOString().split('T')[0];
    if (log[key]) { streak++; d.setDate(d.getDate() - 1); }
    else break;
  }
  return streak;
}

function updateStreak() {
  const streak = computeStreak();
  const btn = $('dock-streak');
  const lbl = $('dock-streak-label');
  if (!btn || !lbl) return;
  btn.dataset.streak = streak;
  if (streak === 0) {
    lbl.textContent = 'Start streak';
  } else if (streak === 1) {
    lbl.textContent = '1 day streak';
  } else {
    lbl.textContent = streak + ' day streak';
  }
}

function buildHeatmap() {
  const container = $('study-heatmap');
  const tooltip   = $('study-heatmap-tooltip');
  if (!container) return;

  const log = JSON.parse(localStorage.getItem('aiml_study_log') || '{}');

  // Build 364-day grid: 52 cols x 7 rows, starting from 363 days ago
  const cells = [];
  const now = new Date();
  // Align start to Monday of the week 52 weeks ago
  const end = new Date(now);
  const totalDays = 364;
  const start = new Date(end);
  start.setDate(start.getDate() - (totalDays - 1));

  // Build array of dates in order
  const dates = [];
  for (let i = 0; i < totalDays; i++) {
    const d = new Date(start);
    d.setDate(start.getDate() + i);
    dates.push(d.toISOString().split('T')[0]);
  }

  container.innerHTML = '';

  dates.forEach(dateStr => {
    const count = log[dateStr] || 0;
    const cell  = document.createElement('div');
    cell.className = 'heatmap-cell';

    let countKey;
    if (count === 0)       countKey = '0';
    else if (count === 1)  countKey = '1';
    else if (count <= 4)   countKey = '2';
    else                   countKey = '5plus';

    cell.dataset.count = countKey;
    cell.dataset.date  = dateStr;
    cell.dataset.raw   = count;

    // Tooltip on hover
    cell.addEventListener('mouseenter', e => {
      if (!tooltip) return;
      const label = count === 0
        ? `No activity on ${dateStr}`
        : `${count} session${count > 1 ? 's' : ''} on ${dateStr}`;
      tooltip.textContent = label;
      tooltip.classList.add('visible');
    });
    cell.addEventListener('mousemove', e => {
      if (!tooltip) return;
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top  = (e.clientY - 28) + 'px';
    });
    cell.addEventListener('mouseleave', () => {
      if (tooltip) tooltip.classList.remove('visible');
    });

    container.appendChild(cell);
  });
}

// ── Fuse.js search index ──────────────────────────────────────────
const searchItems = [
  ...MODULE_META.map(m => ({
    type: 'Module',
    label: m.label,
    tag: m.tag,
    desc: '',
    execute: () => openModule(m.file, m.label)
  })),
  ...PROJECT_META.map(p => ({
    type: 'Project',
    label: p.label,
    tag: p.module,
    desc: p.difficulty || '',
    execute: () => openProject(p.file, p.label)
  })),
  ...LANGUAGE_META.map(l => ({
    type: 'Language',
    label: l.label,
    tag: l.tag,
    desc: l.desc,
    execute: () => openLanguage(l.file, l.label)
  })),
  ...CODE_META.map(c => ({
    type: 'Code',
    label: c.label,
    tag: c.module,
    desc: '',
    execute: () => openCode(c.file, c.label)
  })),
];

const fuse = new Fuse(searchItems, {
  keys: ['label', 'tag', 'desc'],
  threshold: 0.4,
  includeScore: true,
});

// ── Init ─────────────────────────────────────────────────────────
function init() {
  // Apply saved theme before any content renders
  applyTheme(state.lightMode);

  // Jump animation on very first load
  if (!localStorage.getItem("aiml_first_load_jump")) {
    const themeBtn = document.getElementById("dock-theme");
    if (themeBtn) {
      themeBtn.classList.add("dock-jump", "dock-reveal-label");
      setTimeout(() => {
        themeBtn.classList.remove("dock-jump", "dock-reveal-label");
        localStorage.setItem("aiml_first_load_jump", "1");
      }, 2500);
    }
  }

  buildNav();
  buildProjectNav();
  buildLanguageNav();
  buildCodeNav();
  buildWelcomeGrid();
  buildProjectsGrid();
  buildLanguagesGrid();
  buildCodeGrid();
  setActiveTab(0);
  updateProgressBar();
  updateGreeting();
  updateStreak();
  buildHeatmap();

  if (window.location.hash) {
    const hashFile  = window.location.hash.slice(1);
    const match     = MODULE_META.find(m => m.file === hashFile);
    const projMatch = PROJECT_META.find(p => p.file === hashFile);
    if (match)          openModule(match.file, match.label);
    else if (projMatch) openProject(projMatch.file, projMatch.label);
  }

  hideLoadingScreen();
}

init();
