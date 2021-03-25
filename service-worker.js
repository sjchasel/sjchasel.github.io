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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","8d33bbd85bdfdbfee0925efd39643c53"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","3e89ea1f7f625a4a3a3f94fb3bd91e6f"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","94338dc7ba4aa3a3274fd50b9bed271a"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","048f1ae481c1ecc33811b0944dded3e0"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","a31938a0bfb4ff9d75e5c04ecd33a223"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","015351282546bf038326b96401bc7e59"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","0e63530633f2dbf88a5ccdc083c7a3c0"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","c5ced34c009dca4be9e57eed9ae49197"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","8a69debc6cf1ba69b807c559a677259a"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","2fe807b0c07cee2c6167e3112e85d7ea"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","b21346b8fa108d32422a029cfd769cdd"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","3c8dcf1ee3ef241dc09ec0a35b6cbf0f"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","7830611cbc2bb2b3e73bd0fbf812375f"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","59fcd3458ae712a9b77b62f5d0b3d167"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","5c11826010e2bec6caa1e7081553c5a0"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","d43823e841455e4a6f34ba710a33f380"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","09d6899440951a2cadd465020c71fe06"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","3862312ede43f13eb22bcdbbe0efb460"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","ad3c98480f98c1671f0ab18e628a331e"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","bbf9bf0dc53ce5e0c908c73d847d1122"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","4226ac81238d9ac715b8a4856c249658"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","46fb68ec7ee53ced25cb86fae409f6fa"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","c4d7c315387451d7c493792d4d341a8b"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","de3f4a20dbf99c1171092b0a831c286e"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","9cf2e50f593e83ac6dc621a8ca2c78e7"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","76ebe93d85af95b783c28e24dbeec454"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","e6cf6f784a8b7f611a60c417a6e27daa"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","359a89082113e5c748b8d32572f73723"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","fdad862510baa3983e90f81c02b95fff"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","bdd920792f5c05fdaf3b9473c10c0fa6"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","b8fe56d28a1150c8af5cf3fa1ccb2687"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","f3a013a2a5d3bcc93a3c5f508b7dbdd7"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","911d90a9c7374de96581bea98bc3961b"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","07c4c8e56d20bc905ff08d5e8fd77779"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","5fd059614832f33ac4281f0229260363"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","9bc4539124ea88c98a7e5a301542a5de"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","ceeed58bea95cf3916845d3b3d30cd6f"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","b6b18ea03b0aa8edfe26dba680573ef2"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","357a17fa66dbada21b1a09fb171c0c3e"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","be3a91af1e16594a684aebf787992a2f"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","9a20c93fe29638416b3785f9e10c16e2"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","3dbca3ebb491a519902d14b8a90e83ca"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","c4faf65f1d943c67ca9226e72fc9686a"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","04dd4f017759eb1cae35c183b8cef0a2"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","3bcef401abd1bb34d8e2f22df49da5fc"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","7946fc40eb86498496b74c43e2f84b7a"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","083f1e8805f6891d8492efbab288203b"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","c1345972c1f402c26a3bcfe6e37b0e9e"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","f6d89af2af823f25dddd352388cbda10"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","17b88d2d0ded70350f0aac15c06e880e"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","c021e5d8a22607f4d96cd40c1bc4402a"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","b7bc4daecb4283a0bd9f035ee3ac9446"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","1e9c69ff4f1f707f0daa4c2117bd052d"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","d9c7116d89e1374b99261e22787f935e"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","ec33e8a3637cac2cfd8704fdf9fc3e0c"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","5be6ef9ba84d990a7378a57b0fec7a29"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","43128232bee76d910c0698f8e60e4e39"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","0eda2439a2d3f5ccf138f9908c1e466b"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","ccf1f973f60ae4a9ba7e390f3e1eb10a"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","677560a1bdfcf31fa05f44f315dff2ff"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","001b13ce8035e3c966c9eab560b609b1"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","2d24dbea6fa9444c68106747047aaa5e"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","d5cd4195196c1cc83765523a7fe9f923"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","846f1379abe0f5d182ed9c85ec9f8985"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","5089988aa27835769211fccfb886e8eb"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","e961b820b6f7604150cf2b10ed4026ec"],["E:/GitHubBlog/public/2021/03/20/20210320-0326总结/index.html","b35c9dbad623af4732cbabdd589bcd06"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","c8ffdcf86ccc1ea4678e14bf10cc17f2"],["E:/GitHubBlog/public/2021/03/23/DeepKeyphraseGeneration/index.html","04a4c8fac8973a5049413c796a2bb525"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","a6354d0072db362b7ec46ac14ea181bf"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","fda16d85c9fbf25a2853f787eaf9c6bf"],["E:/GitHubBlog/public/archives/2020/01/index.html","ae43c09c5e8f533fe6f6e6b40677e033"],["E:/GitHubBlog/public/archives/2020/02/index.html","88dd946228981bf3dfe8cf65c99cb259"],["E:/GitHubBlog/public/archives/2020/03/index.html","3cfd0e9fbfb79c9f8007178c9f3af671"],["E:/GitHubBlog/public/archives/2020/04/index.html","a7f78514c7daa0375cf3257f31ce220f"],["E:/GitHubBlog/public/archives/2020/05/index.html","8fc3f17db3860ced1d29e3ab7ff30b2d"],["E:/GitHubBlog/public/archives/2020/07/index.html","d969921f9ab93c53f8332c6b7f674b6e"],["E:/GitHubBlog/public/archives/2020/08/index.html","7c626770ccaa199f1cd15a4ae9368f56"],["E:/GitHubBlog/public/archives/2020/09/index.html","be21a7de79efe9a53aa99586009f27af"],["E:/GitHubBlog/public/archives/2020/10/index.html","ded15b53ea5d65efa0673c7c2b10633e"],["E:/GitHubBlog/public/archives/2020/11/index.html","17e5535fa483dfaa0bcea562404e00e6"],["E:/GitHubBlog/public/archives/2020/12/index.html","72a7cf37997a3592e83b185fb560046b"],["E:/GitHubBlog/public/archives/2020/index.html","ad5a1a00a6667a81581b44d1682eeff0"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","7f1620571e9332db4a10d0433b236105"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","d3601d376da784ee8cdb62b71d38d63e"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","ada963809645e25d58f488eb1203b573"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","27cada835226732df45e64051c0e8f38"],["E:/GitHubBlog/public/archives/2021/01/index.html","0a1b8e0adae2c3498eb135710160c911"],["E:/GitHubBlog/public/archives/2021/02/index.html","af6c839903bf25c29a6cc63365182257"],["E:/GitHubBlog/public/archives/2021/03/index.html","f8ec9ff70f76244d20b90edda2f2cf1b"],["E:/GitHubBlog/public/archives/2021/index.html","50c9da00df4849fdcd2ca8d04a1bda6c"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","8a8b04a9cb653bcf96f9f465b55a1ac2"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","88c856e8935b9ba6d4cad28eae2f493b"],["E:/GitHubBlog/public/archives/index.html","5a6023c52cc86d2e19d018d366835c48"],["E:/GitHubBlog/public/archives/page/2/index.html","c2a95debf01ac1c1edadd8a07563a0c0"],["E:/GitHubBlog/public/archives/page/3/index.html","05dfa5bd9160a1f71e7b3020a16d019a"],["E:/GitHubBlog/public/archives/page/4/index.html","f44d6546377a495939854b64c6b250d7"],["E:/GitHubBlog/public/archives/page/5/index.html","afaa6c2f38cf14c3177e0a20546076fe"],["E:/GitHubBlog/public/archives/page/6/index.html","baa9e9d60f246491f57ea0115489800a"],["E:/GitHubBlog/public/archives/page/7/index.html","d5e95c59f0f772b8510868cbaa31d4e8"],["E:/GitHubBlog/public/archives/page/8/index.html","261d1bf43ab9f753eedfd75b4175fd30"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","41d3d332e0b9d26dc5593a00c53b34a6"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","4133c431dd9e2eb23c766f433346613f"],["E:/GitHubBlog/public/page/3/index.html","059c34efdde758bb627d1990ad2d103d"],["E:/GitHubBlog/public/page/4/index.html","d268cbcb8cadcb5c4600984860ca0a0b"],["E:/GitHubBlog/public/page/5/index.html","782136f355236ff79376892231b5498f"],["E:/GitHubBlog/public/page/6/index.html","7575eff910ecce5bddf52691aa325a02"],["E:/GitHubBlog/public/page/7/index.html","a492addfb0f02dc169170258d8632d0b"],["E:/GitHubBlog/public/page/8/index.html","eb5425a8b0b1852e9c6bb004bf4d14a9"],["E:/GitHubBlog/public/tags/Android/index.html","6c5470b23b7c2e7cb13a48111228729e"],["E:/GitHubBlog/public/tags/NLP/index.html","aaa7d5fc57132c424f6004547f9d1a8c"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","c13de61f21030a184707e85de2d8240c"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","4b85ed1ee41fdb3a524b6ccdb8409826"],["E:/GitHubBlog/public/tags/R/index.html","73ce3da77448ac218a085aec29bf9975"],["E:/GitHubBlog/public/tags/index.html","48efeb9b7297b3c51ca8ed478fae41f4"],["E:/GitHubBlog/public/tags/java/index.html","3b6571e223e53da0485636dce6f7922b"],["E:/GitHubBlog/public/tags/java/page/2/index.html","c61441d0e5b80b6731556d70f103bb40"],["E:/GitHubBlog/public/tags/leetcode/index.html","1248d469f62344c2dd124765c9790cf2"],["E:/GitHubBlog/public/tags/python/index.html","2cd317b59a6c9dc73a240ff693e5cb68"],["E:/GitHubBlog/public/tags/pytorch/index.html","854c6e76236cc5ec2c7baa67f465c067"],["E:/GitHubBlog/public/tags/代码/index.html","38a2ca46d813b1818d1f03ec7b476d59"],["E:/GitHubBlog/public/tags/优化方法/index.html","f9ac7988699e4d433c48c9526307e563"],["E:/GitHubBlog/public/tags/总结/index.html","c17e58889c973ee720b327feaebacb55"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","3a8c23298c6acb695629bd01130d66bd"],["E:/GitHubBlog/public/tags/数据分析/index.html","a00c1b4d7e5f4a8524e791f712d0fd6a"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","3125e2a9f32ec1dfdba1394734020764"],["E:/GitHubBlog/public/tags/数据结构/index.html","abe1119891d53112a091ea311c386e2f"],["E:/GitHubBlog/public/tags/机器学习/index.html","18099ff075cff09a34ce1bbb551bed7a"],["E:/GitHubBlog/public/tags/深度学习/index.html","3944d699a0db551a75225c9ded3cee74"],["E:/GitHubBlog/public/tags/爬虫/index.html","975c0863757f1ef48417608844cc9e56"],["E:/GitHubBlog/public/tags/笔记/index.html","16a70a5448a128f51d8ebd2ba74f571b"],["E:/GitHubBlog/public/tags/论文/index.html","4e8e6c4c9583af20b587b4328799cfcd"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","1ea7fa8de102b48df5885614ac140830"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","ca3d0cd8852aef31e8a8af4c2f21af97"],["E:/GitHubBlog/public/tags/读书笔记/index.html","25ca11626ced57b35e5a88b10c56e2c9"]];
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







