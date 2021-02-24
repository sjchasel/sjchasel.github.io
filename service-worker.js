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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","7e68b8e8942445a78175dc36d96d4699"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","93909376aa9832b56f634bf5ee3f4d4e"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","ff7af11e4ec7e58743bb8221511ebb38"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","6bd40a008870ad11ad0ed159ea671cda"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","ceaf8be1c8eb13bc171f317cb86915da"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","68edb8570922255df5616ac9c676c5cd"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","053d7c32242e05fadd3da09a954db3ed"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","41216ec5fe18e29b76950c97b3f0f4e8"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","dca707ba1cd0c3a5789575b5ca40c10c"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","30d9289079d42f96d079b46a9a09e782"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","bda68753b43fc912a9e301baf505dcf4"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","1dd0467d98993791eedf9230a4d589d8"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","5416249cdf41fddbeccfb65aa3de854c"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","d9f73f8c721665bb47111387412ead99"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","94a4f05744f86632ea50cb5417d8186f"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","9e2fb51b61b291fdbebc75326cb31a2a"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","33626919b5c5478a270d5bf648e7813f"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","153385b3553010458999bdb0e4a79074"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","34fd3e7c27e0e18cf19c7846345cc851"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","436ac3eb94fd66213c8c38dd70206303"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","30141d0ebe6e98b88a30227dea4a080e"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","e415ff3a4ca3ae306c7e833ae325b2d1"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","34a5a665a979b59b2f1543e6b889d900"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","0084b86f128ee3f79df997342c5104a3"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","45361b48ca5cb55d1bbd644c2a742604"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","a5c9439878c951738eb6c0803dcf2be4"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","a45cc856a02678921cf92942cc32d09b"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","887b4db06a4460019741ba533ea4e003"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","75a5a658368a69392cc573601a13d407"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","1e51958762b5e0d648dad3969d03f593"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","ad80035ad0ace139542b5de69190ad15"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","36fb29f8d05d54cf8dcf37d4553fd032"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","e0be11fa66562b6d926af69df6b4c2fc"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","3e7e37d1d9630ac8367176bc5f683155"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","36f7f3fb8fccb8a7921cd8e213949c15"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","0ce1ab03c3bfcd9f89bc143d28184cd6"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","0dde8337aa8beded911d9bae616d84e5"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","eac2182b94bf88af38f63d07bd45b431"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","9f0289f09381c5ae3fc6092baa68c9b8"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","1422248923dac50a6946602f92f42bcc"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","752d0577ba1c80de3475a90afe109206"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","d85664ef15743cd830784392ab5b5dc9"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","b1bd14e19f947dd196c88198dd1f18c6"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","e75e7863b54be45f45d0ee4b52408328"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","778a7b1e6c48aa92d696bc45414fe4a7"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","d6246d34f669146a3a38480a1935606e"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","5ac72e744b53f391c7cef549cd977eed"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","9b513d7d0ea066e1b2b2af51198b9b7d"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","ab54aa7f46e9a0a17366b575becd1191"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","20d5303c0bd9f7eaddca5730ab3a0b16"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","6be376441f75eb0ab1eb064cb9299ec6"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","a3aca6af7dd9811b25c1e589bf95dbdd"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","ef43a7f71de5a64cd8d2a1ae844e5bdf"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","b371ec43ddf94cbeed82a5d4d5386967"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","3c9425cbdc9273f792af613fdd2ecb58"],["E:/GitHubBlog/public/2021/01/22/代码理论/index.html","65adfa6803898303fd5470fcd1b594b5"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","90099080e7e28aea6746cc362781307f"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","2c4bc3267c05866003cda2b3eec411cc"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","69693be873d25e40964e52a965147f65"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","0588abc34bfd4deae0e39660547ebdbe"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","8c38d284b4e416910be99a7087e551c4"],["E:/GitHubBlog/public/2021/02/20/AcWing基础算法/index.html","16424a4f4854423b99cc3a11fda30cd9"],["E:/GitHubBlog/public/2021/02/20/DCIC共享单车task01/index.html","a1d4d4f8c1c0fff1a5b801ffa087cdcf"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","e1fb2c6c3ef05b66d69fe335f1fd6448"],["E:/GitHubBlog/public/archives/2020/01/index.html","9bc9d6147563512390ec56c76103d654"],["E:/GitHubBlog/public/archives/2020/02/index.html","48d51c62d2f2d64edf7cda1804c1baec"],["E:/GitHubBlog/public/archives/2020/03/index.html","97aec5b12f41740e7b1f3d04ad14fe5e"],["E:/GitHubBlog/public/archives/2020/04/index.html","352b85b5c7c9183d2066f8accec23874"],["E:/GitHubBlog/public/archives/2020/05/index.html","c39362d610baae326f4359102790825c"],["E:/GitHubBlog/public/archives/2020/07/index.html","177a4160424d6c02ec0e5ad8ebb6581e"],["E:/GitHubBlog/public/archives/2020/08/index.html","ab28ddc97449a3928238dd119fb88cc1"],["E:/GitHubBlog/public/archives/2020/09/index.html","48c9dee422dbc6eac2a88d0b513ce963"],["E:/GitHubBlog/public/archives/2020/10/index.html","520c1e36f372d1066d63e782e2a26cdd"],["E:/GitHubBlog/public/archives/2020/11/index.html","7c9655897ef0472e77624f97073518b7"],["E:/GitHubBlog/public/archives/2020/12/index.html","e2e59d20116fee528016684dca3cc56f"],["E:/GitHubBlog/public/archives/2020/index.html","40478b46432ed2bb9b7cac7890d1fbfc"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","0cf784bcaf80a0b1c3d8d567675b4520"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","7861fcc6a5c0f1672051039b30d09fa5"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","248d4f9679a873e6f471e7588a7e29f7"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","fd967dc68f596cb2b4d4992736cad461"],["E:/GitHubBlog/public/archives/2021/01/index.html","89e600c5b260bec43001c1dda7611a9e"],["E:/GitHubBlog/public/archives/2021/01/page/2/index.html","ceda3ff7e2c86937be6703222f4100a2"],["E:/GitHubBlog/public/archives/2021/02/index.html","f424802f8b19488cea04a98691c2c922"],["E:/GitHubBlog/public/archives/2021/index.html","8ca3fce09e924633c60de307c3b7f7a4"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","678fc66ced1a26ab9f5e28f638519786"],["E:/GitHubBlog/public/archives/index.html","c714f0ccd5a12b3623c58741d3386d5c"],["E:/GitHubBlog/public/archives/page/2/index.html","eadc16d1db8489ff099c6b46c3598325"],["E:/GitHubBlog/public/archives/page/3/index.html","9fe508003156dbc568bd27467f9875f7"],["E:/GitHubBlog/public/archives/page/4/index.html","37854ae3ee8f87ffc4d58c8b1d22c642"],["E:/GitHubBlog/public/archives/page/5/index.html","14409bd633ab432f43aac1fa079a3220"],["E:/GitHubBlog/public/archives/page/6/index.html","8c773f36946565c15d03314c4fb59646"],["E:/GitHubBlog/public/archives/page/7/index.html","f7eaff7b99066ce8a063595aeaee4b46"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","fe185c50ddeafbb72af78a620646be22"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","e79f1a376e342d1e9b0e53e3785050b4"],["E:/GitHubBlog/public/page/3/index.html","d4827f26292ec083a9c7a5a6442cc829"],["E:/GitHubBlog/public/page/4/index.html","2988171fbfc319e209121b909d015ba7"],["E:/GitHubBlog/public/page/5/index.html","258dcf9fc8b5a58d06dd97871dd1ec7c"],["E:/GitHubBlog/public/page/6/index.html","47a247d94c37a6fc80a744355ef4cc82"],["E:/GitHubBlog/public/page/7/index.html","22fce0e03bd3aa5a2d024a34ebbcb57b"],["E:/GitHubBlog/public/tags/Android/index.html","6e0ad4cc80475fc2caab39a49bd4e643"],["E:/GitHubBlog/public/tags/NLP/index.html","3ad117a6124d4e6a6f074acf014ab458"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","13ef85f44b07bcd8f522f3815a5a4a7d"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","28e2e32e1fc3bd0890f24117923f8881"],["E:/GitHubBlog/public/tags/R/index.html","d0b807c1f316d7ced10b2418be311a21"],["E:/GitHubBlog/public/tags/index.html","b5d317257cd5fd2b963a54cc0e738e2b"],["E:/GitHubBlog/public/tags/java/index.html","824be2dc39f475bef39e0282cf598c17"],["E:/GitHubBlog/public/tags/java/page/2/index.html","d3b6d4ac2d850d93e0d5621887f6c52c"],["E:/GitHubBlog/public/tags/leetcode/index.html","230ae512d20fec44e3f138d6233bf357"],["E:/GitHubBlog/public/tags/python/index.html","b2dcecd535c160bfe3f23d4ca050b7b3"],["E:/GitHubBlog/public/tags/pytorch/index.html","5b97ac8ee64b507d7d1bcb087526eb1a"],["E:/GitHubBlog/public/tags/优化方法/index.html","99e19b7320609d3a0e712bb9f4d4ac7e"],["E:/GitHubBlog/public/tags/总结/index.html","d3e6b790c6409f119eda423525193c4c"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","cb78344a5f021a2730b8c9ecf29a0808"],["E:/GitHubBlog/public/tags/数据分析/index.html","c8fa15cc50a205b2915a733fa9962240"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","644976d934fbb5c788bc8661bc470597"],["E:/GitHubBlog/public/tags/数据结构/index.html","aac74391638113c840ca6c21e5c35591"],["E:/GitHubBlog/public/tags/机器学习/index.html","a413f28a4048fb839500ca3bd00d99ca"],["E:/GitHubBlog/public/tags/比赛/index.html","baff47ed6789049bcc9885b46bd845e7"],["E:/GitHubBlog/public/tags/深度学习/index.html","fb0769542f62eef33bb07f69f35ae404"],["E:/GitHubBlog/public/tags/爬虫/index.html","cd5f44b8413565036240d3a82b6f2770"],["E:/GitHubBlog/public/tags/笔记/index.html","b69d6fad7e94f7490bae1c7ca821fbfc"],["E:/GitHubBlog/public/tags/算法/index.html","d947c52ef7881bd7efc395943832a995"],["E:/GitHubBlog/public/tags/论文/index.html","abd7e96c4aa3a8996b228f26488b225e"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","59b5d00bad54335f4091d9a4e935be0c"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","b8df691d2348962f840eaff7ce4f9f1c"],["E:/GitHubBlog/public/tags/读书笔记/index.html","269f48414ff13de2b4390c16d4a4e972"]];
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







