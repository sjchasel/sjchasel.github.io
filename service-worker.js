/**
 * Copyright 2016 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

// DO NOT EDIT THIS GENERATED OUTPUT DIRECTLY!
// This file should be overwritten as part of your build process.
// If you need to extend the behavior of the generated service worker, the best approach is to write
// additional code and include it using the importScripts option:
//   https://github.com/GoogleChrome/sw-precache#importscripts-arraystring
//
// Alternatively, it's possible to make changes to the underlying template file and then use that as the
// new base for generating output, via the templateFilePath option:
//   https://github.com/GoogleChrome/sw-precache#templatefilepath-string
//
// If you go that route, make sure that whenever you update your sw-precache dependency, you reconcile any
// changes made to this original template file with your modified copy.

// This generated service worker JavaScript will precache your site's resources.
// The code needs to be saved in a .js file at the top-level of your site, and registered
// from your pages in order to be used. See
// https://github.com/googlechrome/sw-precache/blob/master/demo/app/js/service-worker-registration.js
// for an example of how you can register this script and handle various service worker events.

/* eslint-env worker, serviceworker */
/* eslint-disable indent, no-unused-vars, no-multiple-empty-lines, max-nested-callbacks, space-before-function-paren, quotes, comma-spacing */
'use strict';

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","d0e8395d25ea875780f8164d6eaead60"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","2b0f85b290f0f29e97c843c4ab2e0221"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","39e9a5444240e812b2e7ef3119efbd36"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","67c5df379a7d05dbb3fa6117c29be83f"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","069d261d5a42a2b1eca0849669896051"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","7d92a6931d20fbd5973f543ba405bcd9"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","f9e5652464080f2c8c90fcae551a67b8"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","a5889d2b1bc0c7dc3fc3c3469d30c6e1"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","550d5844e27da730c0d66a2f6ff01dab"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","f4134e27ff227841cdbad25cae90cf8e"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","d61c32644e2c69dcff26dc0294670a4c"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","75e2b916414d5718a17c16452feea82d"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","7cae81c9a44fba3f2006a8974ea4054f"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","682f9973de843498b783ae4456c778c4"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","5e12d25ca41b10867189547ccb73cfae"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","cf67d9eed9b1cdc096650116d6cbbdba"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","02d75d9cf2923a2ad95e1bb500ec53bb"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","438ece298d60324b0b08f0e321b07ee0"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","118d78dfa9e53dfa7fe1b7bc7e4de539"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","06f7c12802d9a39c9c9332216cfb4ab6"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","c63d858b6ed47a1e3e568fb8c17ae6d2"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","b0f235c3bdf37dfac681bc8d7f483932"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","6f82506197f1fcb95571af45e5b75c09"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","43f83f0ce102c748293bdead984bf2df"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","2c77f3fe50dce5709506b317cba22785"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","52487b5aa1158c25f7b0d2a3543c0ee9"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","2b359333e8cf5f4de04ddbb4e0d501e1"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","69a64a87aaf0b7f6b62d7d3541832081"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","17821c7b27ec20bdec12e44594bdd06a"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","5bc311b8670f0a4021f9a43ea38465a6"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","1abdb3cdb15c4df58a6389e906321337"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","dc6a0294bcf335267843497b7a5c322f"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","53a0dff2da1ca9b879a4bdf0deedc924"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","fc386ed53415e91d9c77c93b3c6ae97e"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","b0b126454c64e9282f277b09d6919c8a"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","79c51b21b70a0b634cab9e3e91162124"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","fb691c17e54f82895f3a4e444a898b1f"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","c9b568b19dfc3b7635da296fb6d6feff"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","ce66bb2dabe74f3d9f804cb1ae55c1bf"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","7589d741fdc6e14eaf2b23e08fd1d3fd"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","2df253c47f8cbdff27c94b12467a7078"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","bb2fe675c57911a45df3cad70d972920"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","94a259c27bc63467b810bdff740930b7"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","8cc35744aae6c728b2239cfc2faee7bf"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","3eb55539cb203fb1bf11ae1e2e2130b0"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","c0ddeadb442d6f8f46f9ecde4d48d626"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","c77d8c2e07e6ace18f91c8ac610b260a"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","e73adce27271fc7f87060f673d5f4e5e"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","7153503818cf673ab693daf55256c140"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","c9c19a36864cf2ea5530fa877076ead2"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","9ee2d91a269d40eba161e756fc56458c"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","fd99897de0bfaff1d866eadafaacc7b0"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","c89f4df82537c98c5f1af80f08de251f"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","534ab511bfb525751abefd2a15399464"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","b2b87c31e5c1fa94771e8e6eb006520b"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","20f9acdc210bfad447c18e70a6ab44e3"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","ff0ee30baddd488146611b5d32561b33"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","a0b46baa9208d347bd20ad8b1da00cc2"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","bf64c17e49a5e969e4a7ef4547ca1ea0"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","f4a8498ea07fa96a3c6b67a70fadafb8"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","5dc32f60830e5424f632c197cb69f824"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","c2c4cca383397ef2a60c970fafe37f18"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","61ae5101ad284ef35ee6d313301d94ec"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","652b59dd28bab863f16f2f506088ef0c"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","120bd46b923e836ebe80d120b8e6486d"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","7e91762e2492c0b03e3f2d1b5756439f"],["E:/GitHubBlog/public/2021/03/20/20210320-0326总结/index.html","12b3939d038fa321dc8ceaa99ee7599f"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","0a1ab8ece5d58f5baa39ecae3eac57b8"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","81646e8a628f0712c273ab0cecb34525"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","98cb04d9ca2c5b831421ad4142a393d5"],["E:/GitHubBlog/public/archives/2020/01/index.html","3858dc21b3c9705377f9e294a9046773"],["E:/GitHubBlog/public/archives/2020/02/index.html","945105fb262d412ef2193b6ed143880e"],["E:/GitHubBlog/public/archives/2020/03/index.html","ca27cd5f1629cb135dbca681facfc020"],["E:/GitHubBlog/public/archives/2020/04/index.html","9d5bad59b49432eda68efccb3228f847"],["E:/GitHubBlog/public/archives/2020/05/index.html","edb767afbea323caf0236084b8c1ebce"],["E:/GitHubBlog/public/archives/2020/07/index.html","d88436e7b9d89961ac0295f16334d0ae"],["E:/GitHubBlog/public/archives/2020/08/index.html","4bf31420fa0443571beafa79003fbd25"],["E:/GitHubBlog/public/archives/2020/09/index.html","b77f503046e11394f2b22da014ca4a10"],["E:/GitHubBlog/public/archives/2020/10/index.html","a14ba28a5960537f282cf6cd83668707"],["E:/GitHubBlog/public/archives/2020/11/index.html","47050ca9b19adebe3f58b32b6009a9aa"],["E:/GitHubBlog/public/archives/2020/12/index.html","dc3f49a5d8f3b2d5c80e1f192ebec571"],["E:/GitHubBlog/public/archives/2020/index.html","70b30276052d7ca2fc84e916ab85696d"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","64b9a268852e09d696ec91c05badacc3"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","78e0c8137d851414d6b4a15fd4670a45"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","339d004b3e40dcba67e78de11dc775b8"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","bb8920734f2ca5c06944311d2f2ddbf0"],["E:/GitHubBlog/public/archives/2021/01/index.html","25f0c9a3f08549d204de7606e26ffb9a"],["E:/GitHubBlog/public/archives/2021/02/index.html","0e97cca8a0ee290dd639e105d70c60a0"],["E:/GitHubBlog/public/archives/2021/03/index.html","a4d29a1bf80993c47746a059b9abefd0"],["E:/GitHubBlog/public/archives/2021/index.html","2682063ffc71c46317a5ede67de86421"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","d4687c052a9b10936720f5172ce80dbc"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","613f39274137fe92fae990ee3ca2c98b"],["E:/GitHubBlog/public/archives/index.html","088d62446b2b78eab1a912971a94b474"],["E:/GitHubBlog/public/archives/page/2/index.html","98497aa309a73b92ba742edfbff57e8c"],["E:/GitHubBlog/public/archives/page/3/index.html","018b2ea09a1870198dbb4edd252b4c9f"],["E:/GitHubBlog/public/archives/page/4/index.html","c2c97c4300dfc42ee468c2f4d9d47ce1"],["E:/GitHubBlog/public/archives/page/5/index.html","25dbba6e2fba9b15c40e083bbc5b6249"],["E:/GitHubBlog/public/archives/page/6/index.html","0715ad2c751aa2a92ac42b8629cbd613"],["E:/GitHubBlog/public/archives/page/7/index.html","83839b7208c69c328ddc73d9a6bf3b36"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","09618e554281feac67289df635a3bad3"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","6738ba0a16bdbd4d41f71ed36f7cafe9"],["E:/GitHubBlog/public/page/3/index.html","b365ccd7752051acf7026c411e1cef20"],["E:/GitHubBlog/public/page/4/index.html","457637f17be12aefcb3f6c82c7670435"],["E:/GitHubBlog/public/page/5/index.html","6b25f61b1394f70ec7f65df4127c4920"],["E:/GitHubBlog/public/page/6/index.html","54e7065f8e2ffef867583e8126f0ab97"],["E:/GitHubBlog/public/page/7/index.html","93b3d898d04b58d9a482563ebd79035d"],["E:/GitHubBlog/public/tags/Android/index.html","e8ebe877356120c75737bede185930b4"],["E:/GitHubBlog/public/tags/NLP/index.html","7a7831014674e6c7e431a70c6f8d7355"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","789ea1b2a82c09ad96f93c01c88f0a5c"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","a2887f2d9a922371ad831fed72771852"],["E:/GitHubBlog/public/tags/R/index.html","a8ffaea8ff4ad6e74f37d371941dccde"],["E:/GitHubBlog/public/tags/index.html","94b03ed3d018fb65b16b3474f30c4219"],["E:/GitHubBlog/public/tags/java/index.html","18ca085243285e3c2d662ab423eb7ab9"],["E:/GitHubBlog/public/tags/java/page/2/index.html","daac482bb1f162fab591a687711b1abb"],["E:/GitHubBlog/public/tags/leetcode/index.html","908408c5a55f4e15f24afa8d227730cb"],["E:/GitHubBlog/public/tags/python/index.html","00d5ec889d12b5fde9eaa5f3bed450b4"],["E:/GitHubBlog/public/tags/pytorch/index.html","c090a917dbd9a31c11d8369b1b924aec"],["E:/GitHubBlog/public/tags/代码/index.html","278009e38f44d7a40c31866575ef0db7"],["E:/GitHubBlog/public/tags/优化方法/index.html","80c2a504af1a83db9a5225608b49a62b"],["E:/GitHubBlog/public/tags/总结/index.html","a1aae1044b5625cc8c8c06ed94f8e9d2"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","0ce62fb7aeaeb2e192641c62affaed88"],["E:/GitHubBlog/public/tags/数据分析/index.html","f0aab4a8673b7f8a8d4ea6cf9f34bf72"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","f83616be5fbdb2fa43bfa9ed4c0b2ec7"],["E:/GitHubBlog/public/tags/数据结构/index.html","66ddaa4389a6903c94a528a8bfadf997"],["E:/GitHubBlog/public/tags/机器学习/index.html","3ff88fd3107b6eeb6eb7aed92a82a953"],["E:/GitHubBlog/public/tags/深度学习/index.html","4ee7fe67ff2808e30f99a8a3b49da041"],["E:/GitHubBlog/public/tags/爬虫/index.html","2957e3243347900d4fc22335e42133fc"],["E:/GitHubBlog/public/tags/笔记/index.html","768535c51e76b7d9723209b8d02ec082"],["E:/GitHubBlog/public/tags/论文/index.html","91660e7806b2ae9bd3f40072a33fde2c"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","f91ffa05856780d4d75181b505e30c21"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","f670cadc1e1b7f7978873bf02c34cd9b"],["E:/GitHubBlog/public/tags/读书笔记/index.html","48900743c3507749b32008fed58251e8"]];
var cacheName = 'sw-precache-v3--' + (self.registration ? self.registration.scope : '');


var ignoreUrlParametersMatching = [/^utm_/];



var addDirectoryIndex = function(originalUrl, index) {
    var url = new URL(originalUrl);
    if (url.pathname.slice(-1) === '/') {
      url.pathname += index;
    }
    return url.toString();
  };

var cleanResponse = function(originalResponse) {
    // If this is not a redirected response, then we don't have to do anything.
    if (!originalResponse.redirected) {
      return Promise.resolve(originalResponse);
    }

    // Firefox 50 and below doesn't support the Response.body stream, so we may
    // need to read the entire body to memory as a Blob.
    var bodyPromise = 'body' in originalResponse ?
      Promise.resolve(originalResponse.body) :
      originalResponse.blob();

    return bodyPromise.then(function(body) {
      // new Response() is happy when passed either a stream or a Blob.
      return new Response(body, {
        headers: originalResponse.headers,
        status: originalResponse.status,
        statusText: originalResponse.statusText
      });
    });
  };

var createCacheKey = function(originalUrl, paramName, paramValue,
                           dontCacheBustUrlsMatching) {
    // Create a new URL object to avoid modifying originalUrl.
    var url = new URL(originalUrl);

    // If dontCacheBustUrlsMatching is not set, or if we don't have a match,
    // then add in the extra cache-busting URL parameter.
    if (!dontCacheBustUrlsMatching ||
        !(url.pathname.match(dontCacheBustUrlsMatching))) {
      url.search += (url.search ? '&' : '') +
        encodeURIComponent(paramName) + '=' + encodeURIComponent(paramValue);
    }

    return url.toString();
  };

var isPathWhitelisted = function(whitelist, absoluteUrlString) {
    // If the whitelist is empty, then consider all URLs to be whitelisted.
    if (whitelist.length === 0) {
      return true;
    }

    // Otherwise compare each path regex to the path of the URL passed in.
    var path = (new URL(absoluteUrlString)).pathname;
    return whitelist.some(function(whitelistedPathRegex) {
      return path.match(whitelistedPathRegex);
    });
  };

var stripIgnoredUrlParameters = function(originalUrl,
    ignoreUrlParametersMatching) {
    var url = new URL(originalUrl);
    // Remove the hash; see https://github.com/GoogleChrome/sw-precache/issues/290
    url.hash = '';

    url.search = url.search.slice(1) // Exclude initial '?'
      .split('&') // Split into an array of 'key=value' strings
      .map(function(kv) {
        return kv.split('='); // Split each 'key=value' string into a [key, value] array
      })
      .filter(function(kv) {
        return ignoreUrlParametersMatching.every(function(ignoredRegex) {
          return !ignoredRegex.test(kv[0]); // Return true iff the key doesn't match any of the regexes.
        });
      })
      .map(function(kv) {
        return kv.join('='); // Join each [key, value] array into a 'key=value' string
      })
      .join('&'); // Join the array of 'key=value' strings into a string with '&' in between each

    return url.toString();
  };


var hashParamName = '_sw-precache';
var urlsToCacheKeys = new Map(
  precacheConfig.map(function(item) {
    var relativeUrl = item[0];
    var hash = item[1];
    var absoluteUrl = new URL(relativeUrl, self.location);
    var cacheKey = createCacheKey(absoluteUrl, hashParamName, hash, false);
    return [absoluteUrl.toString(), cacheKey];
  })
);

function setOfCachedUrls(cache) {
  return cache.keys().then(function(requests) {
    return requests.map(function(request) {
      return request.url;
    });
  }).then(function(urls) {
    return new Set(urls);
  });
}

self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(cacheName).then(function(cache) {
      return setOfCachedUrls(cache).then(function(cachedUrls) {
        return Promise.all(
          Array.from(urlsToCacheKeys.values()).map(function(cacheKey) {
            // If we don't have a key matching url in the cache already, add it.
            if (!cachedUrls.has(cacheKey)) {
              var request = new Request(cacheKey, {credentials: 'same-origin'});
              return fetch(request).then(function(response) {
                // Bail out of installation unless we get back a 200 OK for
                // every request.
                if (!response.ok) {
                  throw new Error('Request for ' + cacheKey + ' returned a ' +
                    'response with status ' + response.status);
                }

                return cleanResponse(response).then(function(responseToCache) {
                  return cache.put(cacheKey, responseToCache);
                });
              });
            }
          })
        );
      });
    }).then(function() {
      
      // Force the SW to transition from installing -> active state
      return self.skipWaiting();
      
    })
  );
});

self.addEventListener('activate', function(event) {
  var setOfExpectedUrls = new Set(urlsToCacheKeys.values());

  event.waitUntil(
    caches.open(cacheName).then(function(cache) {
      return cache.keys().then(function(existingRequests) {
        return Promise.all(
          existingRequests.map(function(existingRequest) {
            if (!setOfExpectedUrls.has(existingRequest.url)) {
              return cache.delete(existingRequest);
            }
          })
        );
      });
    }).then(function() {
      
      return self.clients.claim();
      
    })
  );
});


self.addEventListener('fetch', function(event) {
  if (event.request.method === 'GET') {
    // Should we call event.respondWith() inside this fetch event handler?
    // This needs to be determined synchronously, which will give other fetch
    // handlers a chance to handle the request if need be.
    var shouldRespond;

    // First, remove all the ignored parameters and hash fragment, and see if we
    // have that URL in our cache. If so, great! shouldRespond will be true.
    var url = stripIgnoredUrlParameters(event.request.url, ignoreUrlParametersMatching);
    shouldRespond = urlsToCacheKeys.has(url);

    // If shouldRespond is false, check again, this time with 'index.html'
    // (or whatever the directoryIndex option is set to) at the end.
    var directoryIndex = 'index.html';
    if (!shouldRespond && directoryIndex) {
      url = addDirectoryIndex(url, directoryIndex);
      shouldRespond = urlsToCacheKeys.has(url);
    }

    // If shouldRespond is still false, check to see if this is a navigation
    // request, and if so, whether the URL matches navigateFallbackWhitelist.
    var navigateFallback = '';
    if (!shouldRespond &&
        navigateFallback &&
        (event.request.mode === 'navigate') &&
        isPathWhitelisted([], event.request.url)) {
      url = new URL(navigateFallback, self.location).toString();
      shouldRespond = urlsToCacheKeys.has(url);
    }

    // If shouldRespond was set to true at any point, then call
    // event.respondWith(), using the appropriate cache key.
    if (shouldRespond) {
      event.respondWith(
        caches.open(cacheName).then(function(cache) {
          return cache.match(urlsToCacheKeys.get(url)).then(function(response) {
            if (response) {
              return response;
            }
            throw Error('The cached response that was expected is missing.');
          });
        }).catch(function(e) {
          // Fall back to just fetch()ing the request if some unexpected error
          // prevented the cached response from being valid.
          console.warn('Couldn\'t serve response for "%s" from cache: %O', event.request.url, e);
          return fetch(event.request);
        })
      );
    }
  }
});







