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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","ffcc9722ee64423d2358728d88a60fde"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","6cdbbc292ae1e7abb8bd585aa85f32ab"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","ba593dd9de1607a497cc472e0b992b79"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","6096ea56ec651d337fe6c54516f4f852"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","7b99ef823227dd433354a0866547c94f"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","fa25ba13f681266d53d31f596aa4e776"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","5fb334ab2c212c31713f5779a970b414"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","b74f4ac7f203ddd298778ff4aee9aeae"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","11fdbef54ccbb944908456c6d7ada66b"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","f06e523bd861eab3f56e0246055a8f11"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","d433f64508da4618e503cec0a3a8864b"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","a0b4cceba871e63173295646ae80278e"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","ebee516ee67803c5616e32238a622f31"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","c45f74a958d0995da2eeaa10c8841929"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","08d73b46792388e8e4b212b7921372a4"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","8b59a5f8956952e6a2d1726330171d2e"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","34ccc830a59f740299aaef4f0ccc9e85"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","f51f66b6e3bd4044aa7cd3ff4953f0c1"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","08e9426c3db53f4bcd9b29e40193f123"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","661b41d9a815ef8a0dd30bcc97e46b8b"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","9cb034061d3f321440226447d06e2666"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","964fc00fbf11607ab084bc14e15af792"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","55c8aaf47c2c931636d5d75bbeff8844"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","0b33896f953f2ef6ba4e388636de0968"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","f27d5df7930c1d844641c3d7501c9a6b"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","7578def33d512525bbd4d289e0902da1"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","e4a16c37e9c840e4b28152e0b6c2624d"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","c662b9b50eee4d5538899eb3a34b5bb5"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","50b01b41ccf146309d93d851d33bfdac"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","a6dd1cad4b9c29911bc56650bbbcfee2"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","1030cb58dfbdb76d4db52260e22eb8f4"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","d77b7a88298b3f58e3f9acfe81c38cfe"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","25921c2413f910af1855c0915d36fa6b"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","61aa0691dc4a8d23aba2ff73570ca487"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","b614600999b6dc2c3af6ee5905aeed68"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","0e102992ddedb248b3cdfd0f4e3558ad"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","24116938fd82b8eef88623f2639c02da"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","f46b9e046d53f33e8efe4d0b0baa867f"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","6b3cec5218f230701db8fd765644a8a0"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","3bbd0ba9cb0317fdff8465cf53de00fb"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","b1c798c22b20953d9ec44e3450057bed"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","f01bd51d5863fe64a00e378b33a83541"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","e001b1d5cb64161841e27b32e81f2661"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","f99311c8a8a6f7f52fd0e39908109c64"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","8f4b250894b95e238ae5d7588e6463b1"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","1af93b8fe5d0875f25bb142317b98777"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","1d73b9045b34ffd17518fad0b1e8dcd4"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","288cf55c155d70d4a3a9ed3589fd91ca"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","5e41bc4b510354ca8f7a44006182890b"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","052229a57254489a1e44208127103835"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","88998cc8c006e6844aa07cb77e42f187"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","06c2ecda6f4938e0e0cd0dfb4d9d44b1"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","768925ed797aa3177015356473ed3bdc"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","851baf2ae1466a8671d53d3c2baf3a08"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","58fb5a952717fa644fbb65c7bf8a8466"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","de1de8c185e0096c0ce05f1b71b74498"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","44aaddf34367617284b6c033a5d5f57b"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","a1de5ee6fdbb6850f288027a10e1b5b3"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","2e37a0c2a666f340c4548dc8e9dbf992"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","e36cd288fcb26616beaf0a3953645335"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","1ea136ce0724d6f994d70c04a9f4290d"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","6b45f30c6ac8800f10f4fbdb32a13490"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","fcdc137c006312224c73f8404e51d58a"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","98db79c0327ab7fda94c5f626fc5d6c2"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","bf777c478e72ce092791993e519a9a48"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","881b2fb2d0fc24a2e20b22f5a9017704"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","7501785bd273764cff63dff86448161a"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","8183e3844b946cd722b32c3c0a7cefda"],["E:/GitHubBlog/public/2021/04/01/野生kpg代码挣扎记录2/index.html","c15bc386a740dd71fadbcd07d31f93cd"],["E:/GitHubBlog/public/2021/04/04/python爬虫爬取新浪财经个股新闻研报/index.html","dae7c14fcde656006bd8a1d403daf723"],["E:/GitHubBlog/public/2021/04/04/西瓜书南瓜书 第三章/index.html","9f4f651899490de1449fea5f2ae57b76"],["E:/GitHubBlog/public/2021/04/09/ip被封了怎么办/index.html","345370c317aae8725d1cfaa9e18b8dc5"],["E:/GitHubBlog/public/2021/04/12/修改代码流水账——数据预处理部分/index.html","3894fbb14de5e66fbc3244acecb0a464"],["E:/GitHubBlog/public/2021/04/12/数据分析（机器学习）作业/index.html","17138960c7664dd672e84391f1e67c7b"],["E:/GitHubBlog/public/2021/04/15/量化文本的方法/index.html","0b79ff7d764413458d82a550242ca069"],["E:/GitHubBlog/public/2021/04/17/西瓜书南瓜书 第六章/index.html","6248a2d694466aa5c1c6d103ab67fda1"],["E:/GitHubBlog/public/2021/04/18/scrapy新浪财经的重新爬取/index.html","deb2bccdf3e429199dbbebe6620d46e6"],["E:/GitHubBlog/public/2021/04/19/AcWing代码记录2/index.html","5e37ae60e7035313e357b9734dc2fdd5"],["E:/GitHubBlog/public/2021/04/20/基于情感词典的中文情感分析初探/index.html","9ae09884f7f3709ca79f719d6c3c63d3"],["E:/GitHubBlog/public/2021/04/21/一些量化文本的方法实现/index.html","b1d0c55055398bb704f97d1a40a8cde0"],["E:/GitHubBlog/public/2021/04/22/修改代码流水账——model部分/index.html","04604d845bb6463fe329faeb5d430e29"],["E:/GitHubBlog/public/archives/2020/01/index.html","97503ee76dc295382f43d07807487add"],["E:/GitHubBlog/public/archives/2020/02/index.html","d42282cb447c09635db28b4f5588a71d"],["E:/GitHubBlog/public/archives/2020/03/index.html","d78803ca9e85d3d5238598af4be6efd7"],["E:/GitHubBlog/public/archives/2020/04/index.html","c0a9fec72c9291a7b280bc8d938ebab7"],["E:/GitHubBlog/public/archives/2020/05/index.html","cdebaa648e3fb68cd2bedcc8aec39611"],["E:/GitHubBlog/public/archives/2020/07/index.html","b93f4c9f5e25da59c4d4d186d58d3af9"],["E:/GitHubBlog/public/archives/2020/08/index.html","63693924ca93a0cebb2796bf1a5eab0e"],["E:/GitHubBlog/public/archives/2020/09/index.html","61a3656606ba04cafecfa361122cd802"],["E:/GitHubBlog/public/archives/2020/10/index.html","cd5d778e469ce0d70c7b94962f8a90dd"],["E:/GitHubBlog/public/archives/2020/11/index.html","7ac9d20e2a6ceda0e7330ee07ca02bc9"],["E:/GitHubBlog/public/archives/2020/12/index.html","33b6e4994a2d2d9884f94889187a87e7"],["E:/GitHubBlog/public/archives/2020/index.html","f32cf7e3628a3f63fb795a60cc08be1a"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","f74e9337562b5f618e0921447f4ba85d"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","d9455e06499ba0585b841db8f7755f43"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","4aed90f7f75e92b045a9bbcbb567583f"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","f75f65b6e1a182aff73954503e06f73d"],["E:/GitHubBlog/public/archives/2021/01/index.html","9c7b2c10f173f7ef3113292002d0f7ca"],["E:/GitHubBlog/public/archives/2021/02/index.html","330e91f812909893de4f73f704cb1809"],["E:/GitHubBlog/public/archives/2021/03/index.html","9b5b6b9f53ed3bd4ccd42fdc4cf8d5aa"],["E:/GitHubBlog/public/archives/2021/04/index.html","786a632cc75fb1bf66bdb451e534c8f3"],["E:/GitHubBlog/public/archives/2021/04/page/2/index.html","96e30dada6cd8206a927e2bc284832b8"],["E:/GitHubBlog/public/archives/2021/index.html","29b9e571dce0fd3c8e6556f644759513"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","2522e975ce6685122c8c3820a456d0f6"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","306faf75015132f3a31136c5ed68b1b8"],["E:/GitHubBlog/public/archives/2021/page/4/index.html","900147a650e819c0a207030a1fe7ec41"],["E:/GitHubBlog/public/archives/index.html","f91f304a7f9cea9c4a77f81e8ee39f4a"],["E:/GitHubBlog/public/archives/page/2/index.html","4297ab52ca6c703cc3f64a3297214610"],["E:/GitHubBlog/public/archives/page/3/index.html","c7b8de4b3a7d44eb4f0c03bf7bf027b7"],["E:/GitHubBlog/public/archives/page/4/index.html","a74cae69ab98bcd5571c17463d0af95c"],["E:/GitHubBlog/public/archives/page/5/index.html","f55dfd4b06c44468e80594e0eafac6bd"],["E:/GitHubBlog/public/archives/page/6/index.html","e547b2fcf911ff3d139bc334e8a77ee3"],["E:/GitHubBlog/public/archives/page/7/index.html","a00ee7e5081309ebfa4af62f7a135e28"],["E:/GitHubBlog/public/archives/page/8/index.html","6d6baa0099315f6da54440e4dc8e4bec"],["E:/GitHubBlog/public/archives/page/9/index.html","3b9101d35945053beec87981e1962079"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/categories/书籍学习/index.html","f683cc4dede904b5bb284ca1daffc225"],["E:/GitHubBlog/public/css/font.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/css/index.css","c86084ab63c8ac8d43ecc89e8c98ae41"],["E:/GitHubBlog/public/css/post.css","d746e759ed361785cd87827baa1a6ef1"],["E:/GitHubBlog/public/css/reset.css","4fd030817d0e23aa84743dabb65009a0"],["E:/GitHubBlog/public/css/tocbot.css","e8f0173e7c5216e5359587a88a570b77"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/icon-left-arrow.svg","6aa3f4fd9ad7a807fb4f7ef626dfd0c3"],["E:/GitHubBlog/public/img/icon-menu-sprite.svg","8f7f2745eb92cf88761f378b0ff98ebe"],["E:/GitHubBlog/public/index.html","49d98c5812d2ca154090fa958d804644"],["E:/GitHubBlog/public/js/paper.js","551d4d268064033a6b0625768834ec3e"],["E:/GitHubBlog/public/js/tocbot.js","427555b1fdee580e22f144e233498068"],["E:/GitHubBlog/public/page/2/index.html","654d6bd2d76a10cb1778d6d54877af45"],["E:/GitHubBlog/public/page/3/index.html","733148e199c9bc74c014ae801817fc2c"],["E:/GitHubBlog/public/page/4/index.html","128145cb51ab9440756c5fab9e9062d0"],["E:/GitHubBlog/public/page/5/index.html","d3fb8964bbf0d58a213ca25169f52d9a"],["E:/GitHubBlog/public/page/6/index.html","15f559048313606d8ae37ce824497503"],["E:/GitHubBlog/public/page/7/index.html","c264b767c3872f7eb9040c15755ccd23"],["E:/GitHubBlog/public/page/8/index.html","f26337d1f8ba780d475d218e176c410a"],["E:/GitHubBlog/public/page/9/index.html","26ca4b052bc5ee9a17e50d282d3168f0"],["E:/GitHubBlog/public/tags/Android/index.html","b8dcc0e5cc1ecd8e6a497cc31f857056"],["E:/GitHubBlog/public/tags/NLP/index.html","dbb6ea1dca2f9e8b97ddd7554797e39f"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","9555713e9a4bf20716ffdf8d1dd1bbbf"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","9bbe0356680c17b0328425848e74cc76"],["E:/GitHubBlog/public/tags/R/index.html","a1fee520d0cc991bd66d389c505c08ce"],["E:/GitHubBlog/public/tags/index.html","088bfe0e505de23f7be6d78e85354589"],["E:/GitHubBlog/public/tags/java/index.html","42404f666204e2ba992718dde26b7e82"],["E:/GitHubBlog/public/tags/java/page/2/index.html","89521581d9ae2423b1cd24e99a8a3716"],["E:/GitHubBlog/public/tags/kpg/index.html","d461e0baa315c26afebbae86cd9ba53a"],["E:/GitHubBlog/public/tags/leetcode/index.html","a61bcd1bbf178011b4182a545af4fb1b"],["E:/GitHubBlog/public/tags/python/index.html","60788b053f21b60fa3b17059c44da922"],["E:/GitHubBlog/public/tags/pytorch/index.html","d46686624f3a89bccb4cc9f9328aeb20"],["E:/GitHubBlog/public/tags/《南瓜书》/index.html","826be82d73208e0fcf737d327b2a3bba"],["E:/GitHubBlog/public/tags/《西瓜书》/index.html","d407015bd279b923b3e799ccd52ff169"],["E:/GitHubBlog/public/tags/代码/index.html","dc9661e070a2a740abd6c7ebbb1109bc"],["E:/GitHubBlog/public/tags/优化方法/index.html","1ac9364bcdd579cc6fa030593f31cb91"],["E:/GitHubBlog/public/tags/复制机制/index.html","9d3e0943d7225883d8e09b7e81ddabdf"],["E:/GitHubBlog/public/tags/总结/index.html","73a61886cd5443821e8ff5c2498e3530"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","58c90315d9b72f2ba3a35527d7fcf9a6"],["E:/GitHubBlog/public/tags/数据分析/index.html","e0035211af2703b1a787a089f24c23fe"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","ba5fcc5e9155ab30cd12436ec729e6f8"],["E:/GitHubBlog/public/tags/数据科学实战/index.html","f277200244660b6ac9844bc23f51c44b"],["E:/GitHubBlog/public/tags/数据结构/index.html","fc1b7cb1ae523113435340d892e93783"],["E:/GitHubBlog/public/tags/机器学习/index.html","9afc6d4aa8efceea0f53f10818876828"],["E:/GitHubBlog/public/tags/机试准备/index.html","15ba2e916f01bab79e19cd7bd694fa31"],["E:/GitHubBlog/public/tags/深度学习/index.html","fd1b3db5c5a0318e88c77d9b3efc7c15"],["E:/GitHubBlog/public/tags/爬虫/index.html","621846bafbaec6ecfcf64b379b9dc47f"],["E:/GitHubBlog/public/tags/笔记/index.html","71b3124d5baa1030b8eac552033d9760"],["E:/GitHubBlog/public/tags/算法/index.html","040e837ccdaf603b6d7a186412b04cad"],["E:/GitHubBlog/public/tags/论文/index.html","bd201d4abba09b92f1c272dba223fa9b"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","8626096f366bd34b50266ea418c9b7a4"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","f53c4a5ebfd2fa01e699ccb35434a58c"],["E:/GitHubBlog/public/tags/读书笔记/index.html","931fcc92e7ba07ff30bf5f1895762b9e"],["E:/GitHubBlog/public/tags/量化交易/index.html","465f9ca1750339ed451cddc0267ab311"],["E:/GitHubBlog/public/tags/项目-量化交易/index.html","0996d8b017b244d2233915c6c9d99bfd"]];
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







