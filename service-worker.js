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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","f6f83ef3d59546695887b5aa5f6e1cc7"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","2c189ea58de8e8ee10cd5ffe3eeb9a55"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","60bed75f7e08c8d80e6cbddee927c3c2"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","a3d57a381a4f1ae68a788c54e7384d85"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","4e0e46fd558915064e58c2996cb42925"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","f5b5a625c403d12c313e895b031b0806"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","216f1c20a93a47ccbdbc432c76a7a4a5"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","6f2c9c22f0be35df962a491d12c39718"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","ed175555e5e9eb4d361282a61a2b59b1"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","73fbdab11b182a6caa8c30a0bb27b059"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","e833ced838a1ee8eac3fa1ef22cf538c"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","503469519e22938a8efda59a80b6c279"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","28d691306d63f92aaf03cbb2d80c6a9a"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","23f45f02f1183d2fae6f589ec0353807"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","094c8d1a4d129a533f5b78f457f32a7f"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","657257249c24facb8cee7a521b516109"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","f29e5ea082b4cf32a91d1cb126239bf0"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","fea189bae4b7a00ce48ce8b05fb77aaf"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","32d6062db41aaa9b2aef75c487e08446"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","9e62b32ef9276b80d279099744e32ed0"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","90805128544432726b71182d3840de60"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","eec6c6f739724f50a72ad53abb9bfb21"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","dcf31eb770c5d016bb26b7ed549bb347"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","7ea88c8ddb8bffb95e0b26bd7df4c8a7"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","0cb4410a47a5eb9cd6755d06d10297bb"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","589d3ca8d479a5d11ccf177a12690d2d"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","fc003f121d4041a8dc57a27fa4b9fd83"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","500c35628027cc081bc8b9c49521b494"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","65261246484664b1a5a4eb7fcb9cfc01"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","ea753b6fe8f988f114a44ac9ba4ed779"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","7aa64d99afbbd222e671141674810d90"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","b6e219cb88c2bec81a8d72e3f177b185"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","6bff4b6e930047eb2fcfaabef67218a2"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","39bdfcffd21b85ffe914cf5cdfaed6af"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","333277d1cff036012170847b87f6f3f9"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","2a527388e5aaa5891471a4feb4eaa4f8"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","efc9e23b2844dc377c1270d0e8b3a89b"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","124f00262e1133685493a9eff85acad0"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","21b576232273510339b7acf86fa19fea"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","3ad7ddf8bd930836c4d3b1612f2edd3a"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","679ffacd705878f9d9f822ea2c61081b"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","dda531bf9bb37ada2bbe3b2d6ce89ee5"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","cfed76e6d6aed53e229cad533d706d2f"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","52575ccfddf7ee47ac4c16f4fdf57928"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","5d077388471778f5ad419c6759b5d6f6"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","6a891934d95725eb7d1c0922d69350dc"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","1a1c6f78b04d3bc290b94087cb3eab0c"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","fc1b1deded946a655550040bc3389966"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","bfa1c2d1c568aa2241481a327064adcf"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","3fc8e4289943fccd83770976e87189f0"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","a99c9ce475ce72b8978842c93e1fdae5"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","eb45bd2bf8741b16c4c1f9bb8a733570"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","97cd7126f43148e9540620c14acbe680"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","360067d0250a6a0c998a686e2428cbed"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","acc87e827ef08fabe765f84678182a50"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","17d8c5d2de9adc5f5c46dcbcda652731"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","dbbfbedd0f70f60d7a69323c3f597ca7"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","8a3ed43bad14919c9332fac23ebe0ff5"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","fc26426e746a553b9bc92f5d0e938d58"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","d50e824d00b2dd34807a551f6ad24f1a"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","5e84aca705e0efeb5f968b52b9f655ae"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","e1d68969fda8bede5516121284eb19de"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","ef01363051c54697aa435515b18612a7"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","2c820113b32bb3b033d816bf08aaa249"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","787fdac4e369cf960ffd4c75fefaaaa0"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","e5a438203c4972109c295288f1c4b177"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","f313c09e86085f4e808eac889551aac0"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","4cf6597e5cf8097b30032eb8ff00f75a"],["E:/GitHubBlog/public/2021/04/01/野生kpg代码挣扎记录2/index.html","8581ca40aa8bcad5bc6c07b40cea0bc8"],["E:/GitHubBlog/public/2021/04/04/python爬虫爬取新浪财经个股新闻研报/index.html","a8ca2e8d0eb3a0718857083646e3a6d8"],["E:/GitHubBlog/public/2021/04/04/西瓜书南瓜书 第三章/index.html","00be22d01c4c6b57efb508843cf45d94"],["E:/GitHubBlog/public/2021/04/09/ip被封了怎么办/index.html","21a5171513d5da54e7bdd0c2dc675e4e"],["E:/GitHubBlog/public/2021/04/12/修改代码流水账——数据预处理部分/index.html","73b6c1bd466bf930fef38733095985ff"],["E:/GitHubBlog/public/2021/04/12/数据分析（机器学习）作业/index.html","6bb45a251eb9b820373cd8691c0a603a"],["E:/GitHubBlog/public/2021/04/15/量化文本的方法/index.html","d91e927d92414a883ea2a80f77ece4b6"],["E:/GitHubBlog/public/archives/2020/01/index.html","1d72ee7a9d897132b8d671c35d37d51a"],["E:/GitHubBlog/public/archives/2020/02/index.html","c7be36084cc335a66465410f7c4a207b"],["E:/GitHubBlog/public/archives/2020/03/index.html","f18429bef5e7606d264bcdc5ab06ab6c"],["E:/GitHubBlog/public/archives/2020/04/index.html","598e3e86cdbef5e128dede6409b41302"],["E:/GitHubBlog/public/archives/2020/05/index.html","0bc3ad75e0094b346aa8a27f42f1239f"],["E:/GitHubBlog/public/archives/2020/07/index.html","f2fe25205ecf9759f69d6d4eabea0d10"],["E:/GitHubBlog/public/archives/2020/08/index.html","a5048ea34caa0305f98065bf2661ffda"],["E:/GitHubBlog/public/archives/2020/09/index.html","fc219fb7b9363b19c5549d01bfd56da1"],["E:/GitHubBlog/public/archives/2020/10/index.html","145d20333fb4cade918fce838927ffac"],["E:/GitHubBlog/public/archives/2020/11/index.html","d518a7af2b4f4619b2e18333dcc848c3"],["E:/GitHubBlog/public/archives/2020/12/index.html","7aa7288ebdae8871c1160bc8cd2fceed"],["E:/GitHubBlog/public/archives/2020/index.html","c56281fca6447df9b845386d0386d5dd"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","6b985e364105c4209aa183f012854e30"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","4fd90c5e0647f3081600c099ab784f94"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","ebefab86db990baa225a4a78195e995b"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","5b28f1325f1ef30b051f277c110fbd8b"],["E:/GitHubBlog/public/archives/2021/01/index.html","70d8d598766667d3c520a70db10ff918"],["E:/GitHubBlog/public/archives/2021/02/index.html","a3e521cad52f422791c3c448daa2c64a"],["E:/GitHubBlog/public/archives/2021/03/index.html","6c836d31105a4f99356636a8b6ce6ed1"],["E:/GitHubBlog/public/archives/2021/04/index.html","bc8fc37a9bce96ec195a631ca3223a71"],["E:/GitHubBlog/public/archives/2021/index.html","fa55bec821222e277e699369e9c88eee"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","40f7d871552a9678c890219b253d7951"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","02de97fcf544cbc5967dd619e400ebf7"],["E:/GitHubBlog/public/archives/index.html","23ab0449b992af3a4103689cef6652dc"],["E:/GitHubBlog/public/archives/page/2/index.html","c29296afb0dfbfea54e4bacd8fe07bb8"],["E:/GitHubBlog/public/archives/page/3/index.html","bbfcb7cbc23f0bad7e16f9d2ae2dcbff"],["E:/GitHubBlog/public/archives/page/4/index.html","aac057e45861118933c48747a09286b8"],["E:/GitHubBlog/public/archives/page/5/index.html","bd71efb8b600c1948a1eff9afdce030e"],["E:/GitHubBlog/public/archives/page/6/index.html","b8dd4ac8b10dffc0f3af9f56324d19f4"],["E:/GitHubBlog/public/archives/page/7/index.html","8915fbd8ad95fcd400dd61817cf25ea6"],["E:/GitHubBlog/public/archives/page/8/index.html","07fc5ccbdd2bd058683fc7ca83e5bdb3"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/categories/书籍学习/index.html","1581cab8c62b60c6bcd07a5d0dc9274c"],["E:/GitHubBlog/public/css/font.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/css/index.css","c86084ab63c8ac8d43ecc89e8c98ae41"],["E:/GitHubBlog/public/css/post.css","d746e759ed361785cd87827baa1a6ef1"],["E:/GitHubBlog/public/css/reset.css","4fd030817d0e23aa84743dabb65009a0"],["E:/GitHubBlog/public/css/tocbot.css","e8f0173e7c5216e5359587a88a570b77"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/icon-left-arrow.svg","6aa3f4fd9ad7a807fb4f7ef626dfd0c3"],["E:/GitHubBlog/public/img/icon-menu-sprite.svg","8f7f2745eb92cf88761f378b0ff98ebe"],["E:/GitHubBlog/public/index.html","9f9ad51892f6115a78bb793e4ad4938c"],["E:/GitHubBlog/public/js/paper.js","551d4d268064033a6b0625768834ec3e"],["E:/GitHubBlog/public/js/tocbot.js","427555b1fdee580e22f144e233498068"],["E:/GitHubBlog/public/page/2/index.html","c82df1a6f738a333d9c6b9a84a87b772"],["E:/GitHubBlog/public/page/3/index.html","45fedccc51b9037d6cd7b07f5211004f"],["E:/GitHubBlog/public/page/4/index.html","c92ffbc782ff37af93fcc0e52e358dac"],["E:/GitHubBlog/public/page/5/index.html","449642579c3927bcc562f427207fa083"],["E:/GitHubBlog/public/page/6/index.html","ed36243f80ca847907e158a809bccacd"],["E:/GitHubBlog/public/page/7/index.html","3237be1657e2d6cf0ce3dee2e4ee65bc"],["E:/GitHubBlog/public/page/8/index.html","c9981907b590d6c3421d7cdeda36a36d"],["E:/GitHubBlog/public/tags/Android/index.html","953faeae795eae9ee13a371e05f99bb1"],["E:/GitHubBlog/public/tags/NLP/index.html","b8a00687de0ce4648c81c2d945c5b1c9"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","fff4bca719de2cc8b055975f37790fb6"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","b965824860244198f1968e3e71743f69"],["E:/GitHubBlog/public/tags/R/index.html","748d01e5517e4196d8dc60c5254f8b87"],["E:/GitHubBlog/public/tags/index.html","831ea634d59204ca35b78687ceea6bc8"],["E:/GitHubBlog/public/tags/java/index.html","7ac9fa92bf8b766582d6ebb395230241"],["E:/GitHubBlog/public/tags/java/page/2/index.html","79ba02d710a725a5c907dd57e75b18b0"],["E:/GitHubBlog/public/tags/kpg/index.html","0f2928cc23ce0df295e531c7bee5e2a3"],["E:/GitHubBlog/public/tags/leetcode/index.html","d44272e97e3d4be47551e2a1039e1c89"],["E:/GitHubBlog/public/tags/python/index.html","661b67a1108b3f6172feee713fbcc2f1"],["E:/GitHubBlog/public/tags/pytorch/index.html","a29c8a6f0b403ab216e6d5400b834956"],["E:/GitHubBlog/public/tags/《南瓜书》/index.html","a521fdfe5c1fd6d7edaaa28450c98c99"],["E:/GitHubBlog/public/tags/《西瓜书》/index.html","702b71b8d024bf68d784fdb67cf67fe4"],["E:/GitHubBlog/public/tags/代码/index.html","0262f1f9cde29c273dfc722e617b5d5b"],["E:/GitHubBlog/public/tags/优化方法/index.html","eb5022ea6a2cc9b1605ca7f29829d246"],["E:/GitHubBlog/public/tags/复制机制/index.html","6f42e315e4e71234d1d506c21aed2919"],["E:/GitHubBlog/public/tags/总结/index.html","0c72a1d5f79b216cee493b5b9dd21cad"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","2687602a32b2ea7da4a0b1fb84334ebf"],["E:/GitHubBlog/public/tags/数据分析/index.html","80357d9bb04024c8d46eb0dbd8c53596"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","b5c33f43261f8f7f69893f237ef05dc1"],["E:/GitHubBlog/public/tags/数据科学实战/index.html","0f80610c2c4774924d0ff749e3781d03"],["E:/GitHubBlog/public/tags/数据结构/index.html","7d6dfde65cec560b61172737cdab82ce"],["E:/GitHubBlog/public/tags/机器学习/index.html","32373d0b454e53a805be937f1f83aedc"],["E:/GitHubBlog/public/tags/机试准备/index.html","64715ba14b389881d416601bb9d33a78"],["E:/GitHubBlog/public/tags/深度学习/index.html","4a4775152c7644f27c250460ba56e7a5"],["E:/GitHubBlog/public/tags/爬虫/index.html","820d0dc24041d03c5ce90cdc948ba250"],["E:/GitHubBlog/public/tags/笔记/index.html","21773b0db268416d3f829956d03582b4"],["E:/GitHubBlog/public/tags/算法/index.html","e9925de6a16fdd76ce8aef316afec94c"],["E:/GitHubBlog/public/tags/论文/index.html","e90e7eb0578ceba334d5999bc7c40aa3"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","67c9a5cd29f99abbe3c1b658b4fb995d"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","b03864eb46fde325c61256a952ee3d08"],["E:/GitHubBlog/public/tags/读书笔记/index.html","6c809cfceeab10d49aea1af43d50e7b0"],["E:/GitHubBlog/public/tags/项目-量化交易/index.html","8a0daba2e7cd64e398a267f1c345a76b"]];
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







