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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","4ef00f30689c63da3e8bcf63128e9df9"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","4742db2e1bab0e903441a66b923d7eb8"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","7f453d1ff4588003206e3870b77f68d6"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","4ac9018672175693e2e32169b4cd36ff"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","31f450fc90886a6fd44a2be412646e34"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","027f1a52df5b7502bf051974e62ae35e"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","c51c9ac8ac337db1867cf8821139d46c"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","e736bb309b4b91198607078265b3f2a6"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","9ae72c60eadca690742daa9911162626"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","e3b9695e0bf744436c968bd6c6f15a1f"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","ed34a87dd5960a85ef062a06413834f3"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","eee047b1862ba51c185887905fe99153"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","8584a552e5cc7b1ed21783c4b7ac8b65"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","6d906965bacdc35a736aca0e2f6e6938"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","7a8f5541738de2b333e3403dd59d0509"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","f11ac66021fbaa06d0a53114657aff5c"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","ce6a70e75818f1c5669f21b98bb53a9a"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","3176234b600893aa6a3b5e86525b3715"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","7c5e9fa503227ab79655e2dd59b89141"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","dd586e9ff15a9d089c49933d82c1c4f6"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","a8fc480dbf956fd37d9a7ea3d99564eb"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","264cfda6bb9d1c5ffeb5c3c668c0637a"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","0cf9d6245cccada368ae6c152b9ac822"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","81b919101d112cb009db6b02b3f72273"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","1a8dadd6dda9a60ce436bd582ca27473"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","3558c8a2783144e481a359246081df9e"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","be2be2f60ca5afdd52d94df2c1b1da21"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","c85d46dfe8bf60c228d1d7fd95ae3810"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","e9f1297bba7a509cd4ee4a9214d6cc90"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","ec32cc54d4f7082b7e43963b5a6b46a5"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","f156483464a4be2b135c6db3d9a67bff"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","781855f01fbbbc097862ae691c58d476"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","4a67916288491ed690195fe930d4b1ba"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","1015d1c477089b0c40dff5ffccf5bcb1"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","2a748d2d91eb65e74a8fa026da21af17"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","cb2cfb1ab53a2308f65015934d59c325"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","1df61702a4a6b58779e21e127e7ce53f"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","02c95dad5e76688d3f49f0f22c7c03c3"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","f5a6edf457f1606abe876c5048b5b2fc"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","a392747fe46d20313f885e8003978653"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","40ea101cd87e4ab6d628d3c43fe9e5ca"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","ba61a3d5eb0d045f1685716c3acc68b0"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","8ff5424bce9a6dfbf9b43833701cbc45"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","31acff5b109709ec73d4554fa1ed1739"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","2e19d8fac9b6f359dc0626fc3afa85e6"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","981babc797f48fee1634d3f7239df34b"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","3bb53abf494c69defd9b193cc3632721"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","0d41d500488d007a16c0bc0f2ef4576a"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","4cdb0cd8565b686a8a3da02231222a7d"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","cfb8e1206fb231e3fd3bf01e092147ee"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","ae0189c80495cadba56f9acb4c2413f0"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","babdebfd228bfccfa01e2934c1e9b0ba"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","c4dc08f1f859fb53daa7f588864b8bb3"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","61c0e3dc921fe0e3560f9c8a7caacc2b"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","f6ee2f6c8b69b53a0ca72e3431a4e82f"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","960ab3332cd3b9d2afdc4b7f2b898635"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","ea94104532b3ed80dd4904ab5e69bb25"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","fc6164fc27d9a548597e4bb89f5eea6e"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","fd5539d9b2014d79264df3accbe8b363"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","b3cd50c38ff897e3be8774a9b17e63c8"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","900fd85bbdda051ab552ca8c7ce485df"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","cd8a73cc8eb9ef5bb210e31f0e054697"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","dfadde8de8b144368d5356c1d161d002"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","f10c9d353e2b22655b2033ea2807903b"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","51bb71e938475a87e284a7aea895d5ac"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","1c007f9f99ea8adf777f5c647ce8d040"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","3cd6b9d20a56e267c8b0aeea07589f35"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","3ef02e609ea354c9a16b92109e88e237"],["E:/GitHubBlog/public/2021/04/01/野生kpg代码挣扎记录2/index.html","372236de920937df2d85c7cf93bede20"],["E:/GitHubBlog/public/2021/04/04/python爬虫爬取新浪财经个股新闻研报/index.html","90bf454c70fd3641405e33643804f5b7"],["E:/GitHubBlog/public/2021/04/04/西瓜书南瓜书 第三章/index.html","e524a93231038681dce86d17bfa158d8"],["E:/GitHubBlog/public/2021/04/09/ip被封了怎么办/index.html","b256f916195a0d87062ae280ed123a6e"],["E:/GitHubBlog/public/2021/04/12/修改代码流水账——数据预处理部分/index.html","9aaea90ecde126df3a9e4fef37335a00"],["E:/GitHubBlog/public/2021/04/12/数据分析（机器学习）作业/index.html","5e4974cdcf1c15d7cdb67f9d3ba65ebc"],["E:/GitHubBlog/public/2021/04/15/量化文本的方法/index.html","094c47d4d29907a803222eb2ff41fb81"],["E:/GitHubBlog/public/archives/2020/01/index.html","cd993e4235190fcf607c751769450c7b"],["E:/GitHubBlog/public/archives/2020/02/index.html","429a673032e09bb720f334420e07e8ec"],["E:/GitHubBlog/public/archives/2020/03/index.html","3e215c5212a5b88ef75cc1ff4c2a9a65"],["E:/GitHubBlog/public/archives/2020/04/index.html","f66a7d67f62b4ab87100dd306793b222"],["E:/GitHubBlog/public/archives/2020/05/index.html","977638ba3e517bd41ff4424e07efcaf5"],["E:/GitHubBlog/public/archives/2020/07/index.html","ffe963b02ac295020a93c84ed50ac773"],["E:/GitHubBlog/public/archives/2020/08/index.html","bd1f6e80ee4ec3019d950b5782fc0bee"],["E:/GitHubBlog/public/archives/2020/09/index.html","c7fde12cbc69bd642856e3de2ff524b3"],["E:/GitHubBlog/public/archives/2020/10/index.html","492f111339dd26f5354dbc5a8d7a70a3"],["E:/GitHubBlog/public/archives/2020/11/index.html","83a6d0b6e56742ae78ffb6669f84cf37"],["E:/GitHubBlog/public/archives/2020/12/index.html","178461f782ec3be2ef1173dd53a3b763"],["E:/GitHubBlog/public/archives/2020/index.html","ae2c4e46ebb6f7ed5182f64cab3075fb"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","1adfa8b28653021f315e59836bd0509f"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","d6c343bfe9873395d45699667cf39f84"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","3ff0f43e65e926fee67bba296f151cd6"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","cd573a0b2ba5ffa3de89d3d525dd5e42"],["E:/GitHubBlog/public/archives/2021/01/index.html","c28e8a3e1bc5f39200c3fe3236471303"],["E:/GitHubBlog/public/archives/2021/02/index.html","0f5f925854f289c78aa0ec4fa321edc7"],["E:/GitHubBlog/public/archives/2021/03/index.html","c958eed890be7271466ec6bd41fca488"],["E:/GitHubBlog/public/archives/2021/04/index.html","2f97d06ed6fd4922295f5385db37afff"],["E:/GitHubBlog/public/archives/2021/index.html","9dee21c3e9aad9fc02d3ef56f5b7b10c"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","6a1932ed5b4c6fc75a5929f6d30c295b"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","ea67c5b203baecea4f0c11c1d1e6d2ac"],["E:/GitHubBlog/public/archives/index.html","04410e2748fda592f3a7b86bfc3b5170"],["E:/GitHubBlog/public/archives/page/2/index.html","fcaa2823e54b39f573eba5a199731972"],["E:/GitHubBlog/public/archives/page/3/index.html","939df5451161bffcb9170273f75146f9"],["E:/GitHubBlog/public/archives/page/4/index.html","a07c7f608907402990662726b1391153"],["E:/GitHubBlog/public/archives/page/5/index.html","18ca028ff7993fa6c67c7be98df5c000"],["E:/GitHubBlog/public/archives/page/6/index.html","ebd83c1c0a363cd481bbd10ff7539c44"],["E:/GitHubBlog/public/archives/page/7/index.html","0ac85fcadf8fd2e0e1071befc187f377"],["E:/GitHubBlog/public/archives/page/8/index.html","7c2b2850b3f25ca4c8a84786e031fe02"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/categories/书籍学习/index.html","55046d9cbbd219af4a28e3d4e57b4370"],["E:/GitHubBlog/public/css/font.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/css/index.css","c86084ab63c8ac8d43ecc89e8c98ae41"],["E:/GitHubBlog/public/css/post.css","d746e759ed361785cd87827baa1a6ef1"],["E:/GitHubBlog/public/css/reset.css","4fd030817d0e23aa84743dabb65009a0"],["E:/GitHubBlog/public/css/tocbot.css","e8f0173e7c5216e5359587a88a570b77"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/icon-left-arrow.svg","6aa3f4fd9ad7a807fb4f7ef626dfd0c3"],["E:/GitHubBlog/public/img/icon-menu-sprite.svg","8f7f2745eb92cf88761f378b0ff98ebe"],["E:/GitHubBlog/public/index.html","2dfe8471bc67e4031c8d4ea6b33ed97f"],["E:/GitHubBlog/public/js/paper.js","551d4d268064033a6b0625768834ec3e"],["E:/GitHubBlog/public/js/tocbot.js","427555b1fdee580e22f144e233498068"],["E:/GitHubBlog/public/page/2/index.html","a3485b002f27380d1a8ae3b42f870b8f"],["E:/GitHubBlog/public/page/3/index.html","51c4c27fb011e7764f8d0c1c3fc3092f"],["E:/GitHubBlog/public/page/4/index.html","4faee8e7add31637a37655d2c621d095"],["E:/GitHubBlog/public/page/5/index.html","a680ea81077acd6a620b02f829601aab"],["E:/GitHubBlog/public/page/6/index.html","36a884932cdc18742a2eae96bcf25cd1"],["E:/GitHubBlog/public/page/7/index.html","454dea8adfd9d0f24cdbaea3e2b04aef"],["E:/GitHubBlog/public/page/8/index.html","d2e854bb045fdcc44e4f9685a130d1b5"],["E:/GitHubBlog/public/tags/Android/index.html","ffa27b9f54b84626773e2d30a11ebf7a"],["E:/GitHubBlog/public/tags/R/index.html","7ebb70affc52b68c002a830f8624b0f8"],["E:/GitHubBlog/public/tags/index.html","e079234aaa3560cc7d988a1e96ee9de1"],["E:/GitHubBlog/public/tags/java/index.html","e0ac6f6989d15e8b29241e58aac11980"],["E:/GitHubBlog/public/tags/java/page/2/index.html","576c3a94ad0f6e83edc81e0b06b27de2"],["E:/GitHubBlog/public/tags/kpg/index.html","b9c00e4a775ce8273df5e0d0942ce550"],["E:/GitHubBlog/public/tags/leetcode/index.html","495d5405a5759ff4e88243cffef87197"],["E:/GitHubBlog/public/tags/nlp/index.html","5e11acef9401425e625192a593aace5e"],["E:/GitHubBlog/public/tags/nlp/page/2/index.html","f515ed0553099273cddc7f555b2f15b9"],["E:/GitHubBlog/public/tags/nlp/page/3/index.html","5208a4b053b84e4af7cd20c487ca1f19"],["E:/GitHubBlog/public/tags/python/index.html","e06329bf384d2e0a4a4325f008a8c6f3"],["E:/GitHubBlog/public/tags/pytorch/index.html","25ad8a7425a523a875b767f470478bde"],["E:/GitHubBlog/public/tags/《南瓜书》/index.html","be095815c00cc184a2b940910ef6637c"],["E:/GitHubBlog/public/tags/《西瓜书》/index.html","1b47b4a042ac80b5eaeafdea469aee0c"],["E:/GitHubBlog/public/tags/代码/index.html","32dde9ad62430aecc3b1673d62f354ae"],["E:/GitHubBlog/public/tags/优化方法/index.html","e163676d8992d8e34af5ad969164ecf9"],["E:/GitHubBlog/public/tags/复制机制/index.html","050f19e5bce777852fc3ff22f8e524e0"],["E:/GitHubBlog/public/tags/总结/index.html","e68f809af3f587e3b12ce2abecdd320c"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","9999a47fb4ff36485d81dd7a363c4e83"],["E:/GitHubBlog/public/tags/数据分析/index.html","7500b9b4b694c9cc9def329ca39dffdc"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","fc90ef6b2e1bbb479086164d43868af2"],["E:/GitHubBlog/public/tags/数据科学实战/index.html","fa50e871108abd29ae73aba7a7cd6831"],["E:/GitHubBlog/public/tags/数据结构/index.html","b001e7a556ed47d7d50f72d18f274fa5"],["E:/GitHubBlog/public/tags/机器学习/index.html","617a41857925bf0c216ec2032bb86000"],["E:/GitHubBlog/public/tags/机试准备/index.html","af6f74bad8b90ef51f5895ea1dc2a399"],["E:/GitHubBlog/public/tags/深度学习/index.html","5df6336f75a031402858e76ea6ab3973"],["E:/GitHubBlog/public/tags/爬虫/index.html","0c644f7c60fbf91b56576f772716b300"],["E:/GitHubBlog/public/tags/笔记/index.html","8888f53ede5c6c8e6aa7da688db5c66d"],["E:/GitHubBlog/public/tags/算法/index.html","2ed1dba4becacdaf2d3a276113f322dd"],["E:/GitHubBlog/public/tags/论文/index.html","edfa0719b1034b8d808e7f420d31103e"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","e3e7193593dabceacf1018bb199bee4d"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","5fc62a32f58ffe29b436c0acbb65ce4b"],["E:/GitHubBlog/public/tags/读书笔记/index.html","4ad00a27bd54c1f2682e851e84f56d3c"],["E:/GitHubBlog/public/tags/项目-量化交易/index.html","e777e6948fc5e34b623f34fa1ba27e17"]];
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







