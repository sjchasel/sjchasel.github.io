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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","c507a4fc880f9f3a97981cbae130e064"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","c1b2bb11246feee953d561c0d776334c"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","bdf8ed53eb7315a252d7f52518704dd8"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","90f60cb628a16ab6966684c57f616846"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","f7134d6170cb13cad2d4fbf308f8fde3"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","1c19443364b60f12ff2615a6066f0b88"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","461fc312c399c9e538088f47c75e8719"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","552dbae06df8dcd1e238d952452419ea"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","65d43576607a10622757e04d680d803e"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","a6e141a5716d039c18db6bde02ff6c23"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","83e56f6b96e293510fafa42452d0add1"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","c2f7466721aae766bb0e688eef92d6b0"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","cd2dcc565b6f7508e0c48b1a17e53cad"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","319345da3e88c2ec3fca671f487515a7"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","3c7c6ddca2ad5ca1dc79b19764d363ee"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","06595353ac64a5a77c8f56b272b224ef"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","722bf90e5609349b9223a7a88a64ee84"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","8d18b52986614051597c03fc6ba7656d"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","3fdf93a6c69cc113949b0690dd826fcb"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","285544c23a2c1fdf62e922fc2a9fba44"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","67775a74ddd51e8bae7a3598f16c50cf"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","c4d83fdf51a6a8c93c63be4e8fa53a6f"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","38f087df5e9d823d27cabf51dc3a9f32"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","020a59199a8f8e4a9df7bd76c5ae4e27"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","a916f66fe89757a99705f20cdc1237d1"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","ebcdbb6066237cd31ce3b858ec5e6875"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","eae909a3ebee191f3b65731b483e9172"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","5dbb6d7e2350656e07ac08f01343b8a4"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","be008a263af1e845b4c25f6908b060d5"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","947f524a1f78533bcbe5c299be29ed82"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","36ffc65480bf60d90284726ccf63f2a0"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","a42945a8fb8fba685dc641cdd6ae8c7b"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","a26ceef8ac142ab6ef3c6df7ae99a8d7"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","d4de64c4cb191a8bf09e45bd0f5e44ee"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","33e87df3994f11f595018df6a6d24f75"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","d6e5ecc0efee85a72b6d366786d2d891"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","68f573fb612a48d014ac09e47cd5a26d"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","3bfc8fee70e09d0c43a22257737987a3"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","21c62b72a8dfe328ab7c5a832e3c892c"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","76a8b3a0d7ef1f75850115ab8ea2ac98"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","833e6f44dc9ace4d234791449eece279"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","ee7a8936eb7c014988444000a6ecb43f"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","7f8d641c37124a117c331828766e7db9"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","683483d433288e482c71bf137fed5f68"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","de4fd66cae989b3d7a5d1628b4a33eb2"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","d14ed29e476df03f05a67a1e69e7343f"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","8d2a15bc398832cdf1e3a971c19bdb3d"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","dc1192236f4b3f3e31b536c7d7e50673"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","403c64ed4634f49e2f09860498db50b8"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","ff5703fc603c1bb0b8976ebcf1a1659c"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","6c6be14ba813b07262d81a1b39996b0d"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","ee24028ed82ec23bd19f710ce746557b"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","b8c7d76665b8393142fd42222ca719e0"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","0b15e8c7f7ff7c20873b4125050194ce"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","1e35daecd5fbdf02ec317f8ae2693e8e"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","98f10729c83bda091b7e0a801aab8551"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","fca8313ae253c45001db83abd7cc67f2"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","b3f5651997f24ee0ce216fa4c82d2344"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","e974bb3eb6a34eb5685d305d67a5aa04"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","6f356622673eba738d6b510bc098818d"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","640bff0561d0f4077ae73048256efccd"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","cc86e0ffc45d0ff5b66b4933872b0aaa"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","5a1fcffaefea09fc0efe2527a2cdb20f"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","62a9e730619ca7c12667cfd042760871"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","32737eee1fc22f0cc055d6fb320e189e"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","fd4144c64b160f88035e140faf97283b"],["E:/GitHubBlog/public/2021/04/04/python爬虫爬取新浪财经个股新闻研报/index.html","6c7b1ef7eb80ff626363ac21eaf188c6"],["E:/GitHubBlog/public/2021/04/04/西瓜书南瓜书 第三章/index.html","6cb4f7232753776f3a0773b83562b304"],["E:/GitHubBlog/public/2021/04/09/ip被封了怎么办/index.html","0643752a05514d6a4f0a6f65df01e7a9"],["E:/GitHubBlog/public/2021/04/12/数据分析（机器学习）作业/index.html","48de60a03729ebd3e9c35a266d1c64bc"],["E:/GitHubBlog/public/2021/04/15/量化文本的方法/index.html","be75158bb9b1c1d1b44b9f8b7b7b37e2"],["E:/GitHubBlog/public/2021/04/17/西瓜书南瓜书 第六章/index.html","771d3ab221b3bf5f55f9dee151e1da99"],["E:/GitHubBlog/public/2021/04/18/scrapy新浪财经的重新爬取/index.html","ca298c9baa036a8f4f9d343bb48ffecf"],["E:/GitHubBlog/public/2021/04/19/AcWing代码记录2/index.html","cc71653dc1ded0f23da307c08a2e6925"],["E:/GitHubBlog/public/2021/04/20/基于情感词典的中文情感分析初探/index.html","c490f08d802a6b25a9b26d3f257bd87f"],["E:/GitHubBlog/public/2021/04/21/一些量化文本的方法实现/index.html","06bf7714e7efabd4bc57dc98623b13df"],["E:/GitHubBlog/public/2021/04/30/正则表达式笔记/index.html","fc52503b528bbbf75fd9b8a2fb5c4d19"],["E:/GitHubBlog/public/2021/05/02/同花顺量化交易平台探索/index.html","99146180da6b1d4783d63598f2c464dc"],["E:/GitHubBlog/public/2021/05/03/C++语法基础课题目打卡/index.html","e2beb0f64cfd35160226483c011fef3d"],["E:/GitHubBlog/public/2021/05/15/量化交易论文总结/index.html","cc7ba5b339a8c94f0e88f6f6605e0e7c"],["E:/GitHubBlog/public/2021/05/19/AcWing代码记录3/index.html","08e7547401740d3fc487449544757992"],["E:/GitHubBlog/public/2021/05/23/BFS和DFS题/index.html","93ce92b6ca7b10e4c79b145d0439cecd"],["E:/GitHubBlog/public/archives/2020/01/index.html","470722227b7df22d672fc4146d156f60"],["E:/GitHubBlog/public/archives/2020/02/index.html","da44e3619a52cfdc1f89ed575a9a9990"],["E:/GitHubBlog/public/archives/2020/03/index.html","04716aec916fcc5e5e740a89e03615ea"],["E:/GitHubBlog/public/archives/2020/04/index.html","96dcda672b7a775e89caae2645c7ff18"],["E:/GitHubBlog/public/archives/2020/05/index.html","1b959d40ed87d0fcd488e1b375a8fbef"],["E:/GitHubBlog/public/archives/2020/07/index.html","9609f0a4044d85d0786fe9deccf56c57"],["E:/GitHubBlog/public/archives/2020/08/index.html","a8bdbbb63a5333e34376e7e4ade0c1a4"],["E:/GitHubBlog/public/archives/2020/09/index.html","8e7d60319680b1637cda91b6e8f0d653"],["E:/GitHubBlog/public/archives/2020/10/index.html","63ec877f128166d9acf21e1065ebf2de"],["E:/GitHubBlog/public/archives/2020/11/index.html","9ed8e125a4452df9b7da4ddcc97bb01e"],["E:/GitHubBlog/public/archives/2020/12/index.html","aba73fe459c36ae57886290c4fbab96d"],["E:/GitHubBlog/public/archives/2020/index.html","737edbf2afceb85b0cfc8c181dbb7cbd"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","b4dd4ed64e1c6e5fa8facc58cfc526cb"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","245d153d945a9dfed89c4668946f50b3"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","76353594988a6c0a472db1d68027a938"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","9aeff725ead83de8cb2a2f5e1a5092c3"],["E:/GitHubBlog/public/archives/2021/01/index.html","f749b521f80ccff2f090fd190802d4e6"],["E:/GitHubBlog/public/archives/2021/02/index.html","91846efc6160e71a25b9bce512f95b4f"],["E:/GitHubBlog/public/archives/2021/03/index.html","266a19168a7bac36c2e2160a9b99cdf8"],["E:/GitHubBlog/public/archives/2021/04/index.html","b5008e14a750d5d16bba5cced2339074"],["E:/GitHubBlog/public/archives/2021/04/page/2/index.html","03e145b5ec2d704488c09584cb28cf32"],["E:/GitHubBlog/public/archives/2021/05/index.html","1e8978a74e455cccab20a16a14d66ff5"],["E:/GitHubBlog/public/archives/2021/index.html","64a947c2b5c5970beae77c5584a689c6"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","dd8cac9c562bb866f422dd3ead43e6c8"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","c956eff54783b6917b1bd7eba736c1aa"],["E:/GitHubBlog/public/archives/2021/page/4/index.html","cb90b7c6d5436e73d222b3edc5906fae"],["E:/GitHubBlog/public/archives/index.html","f57f6bc37cc47c618ed61d05fb1527c9"],["E:/GitHubBlog/public/archives/page/2/index.html","c6714414be49cbadde9e208a134c7946"],["E:/GitHubBlog/public/archives/page/3/index.html","cd70283a0a29c42559d9e683a28e7590"],["E:/GitHubBlog/public/archives/page/4/index.html","58d0ba491f1dc028d69269e00eebd0b6"],["E:/GitHubBlog/public/archives/page/5/index.html","e3873ffed65f33025c40e498e891c77b"],["E:/GitHubBlog/public/archives/page/6/index.html","f353494040037363a93c035c26648f67"],["E:/GitHubBlog/public/archives/page/7/index.html","b7f5b048e8257686e57fdbb933cbb68d"],["E:/GitHubBlog/public/archives/page/8/index.html","bba0937cd199d04093c6d8b00bb88c27"],["E:/GitHubBlog/public/archives/page/9/index.html","1ce2114e0e1de3253bdd68203eea557e"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/categories/书籍学习/index.html","3f66642e5069daea94f9a8a9348bc229"],["E:/GitHubBlog/public/css/font.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/css/index.css","c86084ab63c8ac8d43ecc89e8c98ae41"],["E:/GitHubBlog/public/css/post.css","d746e759ed361785cd87827baa1a6ef1"],["E:/GitHubBlog/public/css/reset.css","4fd030817d0e23aa84743dabb65009a0"],["E:/GitHubBlog/public/css/tocbot.css","e8f0173e7c5216e5359587a88a570b77"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/icon-left-arrow.svg","6aa3f4fd9ad7a807fb4f7ef626dfd0c3"],["E:/GitHubBlog/public/img/icon-menu-sprite.svg","8f7f2745eb92cf88761f378b0ff98ebe"],["E:/GitHubBlog/public/index.html","d060b3011e123caaa1e14d45eb92e559"],["E:/GitHubBlog/public/js/paper.js","551d4d268064033a6b0625768834ec3e"],["E:/GitHubBlog/public/js/tocbot.js","427555b1fdee580e22f144e233498068"],["E:/GitHubBlog/public/page/2/index.html","eb7d7831507e1dde2c61e50ca5b529a1"],["E:/GitHubBlog/public/page/3/index.html","72db71ce62a61252a703dbbbb8da63e3"],["E:/GitHubBlog/public/page/4/index.html","2aad22f5570107f3d5e83f8c6bfe720e"],["E:/GitHubBlog/public/page/5/index.html","ae44b8023db5788d3b4a481d50fb832f"],["E:/GitHubBlog/public/page/6/index.html","4da1abd6c7c6e945d61aa58566d9d379"],["E:/GitHubBlog/public/page/7/index.html","9949c5f7bd84621569057b217feb1943"],["E:/GitHubBlog/public/page/8/index.html","576eca290fd25f11a10aa3981b092f9c"],["E:/GitHubBlog/public/page/9/index.html","b4fc403350de4a3070ad09f87ef45bb0"],["E:/GitHubBlog/public/tags/Android/index.html","81fcf7188c53ad0e19d7b447dcf62c0d"],["E:/GitHubBlog/public/tags/NLP/index.html","8c05b13865eff56c9732ee0bbd6620a5"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","c1d6742ceecec78681569dfa315fa74b"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","ea7d39335c1de1e17b075f2cd15a0f77"],["E:/GitHubBlog/public/tags/R/index.html","d1ff8d84a33650f25ce9fbb4e3bdf0f6"],["E:/GitHubBlog/public/tags/index.html","72d6ea1bd6c376fcf870b139c90fd942"],["E:/GitHubBlog/public/tags/java/index.html","d28ffd12466334a9ab8f8ade56bbb9ef"],["E:/GitHubBlog/public/tags/java/page/2/index.html","b23c6676411266afc3da717c23de0c7d"],["E:/GitHubBlog/public/tags/kpg/index.html","1e4f730e308780bccc14509e00fc88fd"],["E:/GitHubBlog/public/tags/leetcode/index.html","341143fac3212a01952f39713b1ed488"],["E:/GitHubBlog/public/tags/python/index.html","28b1355f68a1056127ea9492d3d90b9e"],["E:/GitHubBlog/public/tags/pytorch/index.html","2f329db3f1054257f12d3ce26026546d"],["E:/GitHubBlog/public/tags/《南瓜书》/index.html","c2932a8b3beb99300742f087b3959ef1"],["E:/GitHubBlog/public/tags/《正则表达式必知必会》/index.html","70062440f23f35834008198b9778bf2d"],["E:/GitHubBlog/public/tags/《西瓜书》/index.html","b7085f73ef9ca30138b15a9ca411392e"],["E:/GitHubBlog/public/tags/优化方法/index.html","b7ea4bb2f857676b73792ea5102ccd98"],["E:/GitHubBlog/public/tags/复制机制/index.html","380af5183549bef65cefb6aad7e373dd"],["E:/GitHubBlog/public/tags/总结/index.html","695d9d7540a21f90e36d68fb165a055b"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","3d7df5c5c348456b3d5b9be17cdbc905"],["E:/GitHubBlog/public/tags/数据分析/index.html","2018ad3c626b915ea1d3d5799e46730f"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","51d557fcd26641aed938ecf7beb4a266"],["E:/GitHubBlog/public/tags/数据科学实战/index.html","a2fccdbb2973601295c27a2539ff700e"],["E:/GitHubBlog/public/tags/数据结构/index.html","d208f6abb5764a8f385b6a1717548c88"],["E:/GitHubBlog/public/tags/机器学习/index.html","5bad6b31a351e71ccca9d1f4fc433b27"],["E:/GitHubBlog/public/tags/机试准备/index.html","633a490a9ab9752bf5d33c9c74d88d69"],["E:/GitHubBlog/public/tags/深度学习/index.html","6a3b395767a5a294a39538aba318c820"],["E:/GitHubBlog/public/tags/爬虫/index.html","e5d730bf98b06a0ae7d680a134ba44ff"],["E:/GitHubBlog/public/tags/笔记/index.html","6bd5474726dee71d86ad30e8f1ee9a9f"],["E:/GitHubBlog/public/tags/算法/index.html","96552ab2fbc85b1744b4d64645b58b7d"],["E:/GitHubBlog/public/tags/论文/index.html","5592f43784eaa55848ed6955183f24b2"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","eab7a6dd9638fd8a1308de6573404fc9"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","c812260b0e5838a9f90c3f8821680ed0"],["E:/GitHubBlog/public/tags/读书笔记/index.html","35f7b000b841a9d0160c0d5abdf5ab25"],["E:/GitHubBlog/public/tags/量化交易/index.html","e9ed24e7461fdd18929f8f2522668e54"],["E:/GitHubBlog/public/tags/项目-量化交易/index.html","b38f45387a499060d707f311d4e99ce4"]];
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







