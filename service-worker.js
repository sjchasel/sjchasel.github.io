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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","201f1d2797783c628d9a756fd321e58a"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","86509efcab54404a7bfd75ef206ad214"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","7c3b40abee3a541e1eccd49be7aec115"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","92726ceb8da40ed5ba0c47806389b35c"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","f325ed21b6884d86a52df16d645a33bc"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","619d35dce8adfe0a4512a1f7fd1e9d24"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","5ebb394dd46845d8a46dfbdd46c9b398"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","1dedada98548f25ca92f374d2a6842de"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","c437c1944cadc822079a7812b1e52f3d"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","4db17bf06a4b0de3afee675e1147a801"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","32dd265201926d710d0ee6b9e5b922e0"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","67ed4d45b00b66f36b2a4286190984f2"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","bc57379590f64b93a1c725890d754e26"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","046d5ce937e325b3692f1f72aaccb4b5"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","923e748cbbfe82a02646d2128e6643be"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","81d28526a2ec3b6c6dcf66c772b39f23"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","b44474586d7796871214595ea327fc44"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","ec4fa1739d82f429172389fa9cac3732"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","12825e86ece8de364bb942842dc02fb3"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","848009f445052b5fb48f2a14f5255ef4"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","e356c825962eff07ce9fb3f8d7c0d1b2"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","a7acf7c7e8421ac4b2249bd91ca81e72"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","3d1c3bc7a1ba30abb1230bea513803ef"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","a3ce760c28855f3166cca64861bf3914"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","9a26e913d6f70ff48483844df240ee6e"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","674031a7758cab01bb70879112b39bb7"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","ac3ba71c7893bde508f81796a64631f4"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","1f33fc8da87876202445413c9b6008c2"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","179e44d897dfd1dbb73890cbad40916e"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","1163ac0bcd731acfe37099538be2adbe"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","db52364c8aa4835925ea5929999e757c"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","b92ebba55944b93eb26b3269397888a7"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","bc91f2f4905c2dc50f7eaad3feb5f095"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","5914d15f0a0f108bbe7521cef7697670"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","5425f63328d53b8d7529c9eb524f7d03"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","addf72723daa2934991ef93505c6aeaa"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","1e6b4009733b9f39ebbe19e54c6dc07b"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","2e9148d90ce4077fec717e652079b497"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","89d12bdebc6323b2d3660e28d80fb511"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","00f0ef6ff4430ff65854066a2e72fbe7"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","11b050adbced98eb52bce5a795ccd684"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","52b7f4fc8612ff0f195120248b41b996"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","88795e03528d50cc9887e96b98f7282f"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","18c21c284c2bf0485fadde7d1557ce86"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","dffa5b68168b9c4b59701df016c37df3"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","291528fb89b5dc92097601b52d31c8a4"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","8e1cf07c9db5b1a25200d6c5e8fb87cc"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","623b07b1da67687728e4d14190c32e9d"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","335182b6cb2c46d6e4698bd2036602ab"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","56d8c3829e93cf40a8fda00c4975ed94"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","906bbe112991ef94c0613f385b74dc22"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","47850117f1c757dade80f275d26f018b"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","245ea119d92589d130bc8bc62334b5c7"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","d7705252b5f514f6fc5923679e7f0b49"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","2f43d3d7081e98cbb4531e87617084de"],["E:/GitHubBlog/public/2021/01/22/代码理论/index.html","e6bfe6b037a15121ecdb7feb65da4cfb"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","0e94ff1cc5d07d695c705b97530664d8"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","ffa44771709b134141d1fbdec9f6bbfb"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","c4c9491f70bc2a3c0d95770f0a024b0b"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","6a64be59fcaae50449560ff267b7f86e"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","bc1072852648b8828e46839404dcdd86"],["E:/GitHubBlog/public/2021/02/20/DCIC共享单车task01/index.html","c41fb5aa9183625924d492c0630aa497"],["E:/GitHubBlog/public/archives/2020/01/index.html","9d030e8137a71179e02717a55198001a"],["E:/GitHubBlog/public/archives/2020/02/index.html","f45c6781e515fd4dd24b831288e440e6"],["E:/GitHubBlog/public/archives/2020/03/index.html","fc11000113fa36d31979d5c5ff3dc9af"],["E:/GitHubBlog/public/archives/2020/04/index.html","e51fd3c3cfc689540abb41bbda95194c"],["E:/GitHubBlog/public/archives/2020/05/index.html","fa978c25d8cec0d5c0cbed5b9502f673"],["E:/GitHubBlog/public/archives/2020/07/index.html","b11cda0edf02f6664d2f6badbc6a975d"],["E:/GitHubBlog/public/archives/2020/08/index.html","6de610de6f8a72fdc5b66cf0d8ff6f1c"],["E:/GitHubBlog/public/archives/2020/09/index.html","09ba44825c71be7796bf1fe3a2fd49f0"],["E:/GitHubBlog/public/archives/2020/10/index.html","d4430817acfdca36cd5dac7eb6072466"],["E:/GitHubBlog/public/archives/2020/11/index.html","33441b99ae7a9ffcf48a4a1804742c18"],["E:/GitHubBlog/public/archives/2020/12/index.html","3af37c7a0963cab09bb48a98105191ae"],["E:/GitHubBlog/public/archives/2020/index.html","6c9c072266487fead75dff9e686c91cf"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","fca897bdba787e8a642fdd8c96239129"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","bc2da46afe6151802d5d285e8b350971"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","789d424e451c0a6ff16d8c13011d1cdf"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","b361d00dd4f155b35029428d1043217b"],["E:/GitHubBlog/public/archives/2021/01/index.html","765dd73ad6b684f01684a4748e9eb818"],["E:/GitHubBlog/public/archives/2021/01/page/2/index.html","798c3283f59d514b0b3ac428a9103858"],["E:/GitHubBlog/public/archives/2021/02/index.html","ee835768e91b73f44731a67bdfca1255"],["E:/GitHubBlog/public/archives/2021/index.html","a69a1b18066449b693b691e0ce33cbbe"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","e31909df17389196ec6a8aab6921efb0"],["E:/GitHubBlog/public/archives/index.html","8edd99eae1a358b65ab16fca98b26725"],["E:/GitHubBlog/public/archives/page/2/index.html","898a341077db9a69b6c7fcae9695b6b7"],["E:/GitHubBlog/public/archives/page/3/index.html","2a309d53a105d969291134d15e8ca50c"],["E:/GitHubBlog/public/archives/page/4/index.html","f6a613730ce20ed2133584b7fde66b9b"],["E:/GitHubBlog/public/archives/page/5/index.html","c2e87c78daef396eee1171f279caab22"],["E:/GitHubBlog/public/archives/page/6/index.html","09b897d9f5508a58c227f6fc38d3a2de"],["E:/GitHubBlog/public/archives/page/7/index.html","9a9f910daf96180f0ed80dee7c3afc9e"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","0d1641611a7b8337b39609ec36644508"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","1eaf249654c659db74c64d7bf465bceb"],["E:/GitHubBlog/public/page/3/index.html","03321b69c153081c443d2dd500c7841a"],["E:/GitHubBlog/public/page/4/index.html","163ebcdac7c052b3d33e24e3d03d5cac"],["E:/GitHubBlog/public/page/5/index.html","b558c909d5fbab8aa7d28adbe62101c7"],["E:/GitHubBlog/public/page/6/index.html","9f7dc2c25125b1fa5e564f2bfaeeec22"],["E:/GitHubBlog/public/page/7/index.html","9df6ed614851537c6b25eafd6e25540d"],["E:/GitHubBlog/public/tags/Android/index.html","2b0c79a1676ad695b953d26a5249a8c2"],["E:/GitHubBlog/public/tags/NLP/index.html","55adcd8ea2b6ce7d6cce1eea205c3544"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","840070ebda16f6ba3111265cf407b684"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","c1e577d7c385407a2e182383c4d1dadd"],["E:/GitHubBlog/public/tags/R/index.html","f1237a45bf8c40c15967413c8a06374e"],["E:/GitHubBlog/public/tags/index.html","ff33cdc1d8ad700bb672b921d76dae9c"],["E:/GitHubBlog/public/tags/java/index.html","034c9a2f7dd89cff90b27ee3c878cc7a"],["E:/GitHubBlog/public/tags/java/page/2/index.html","30cfa78f8854116cbd8d07e805cfe63a"],["E:/GitHubBlog/public/tags/leetcode/index.html","7e334ddd11a881e55ae8869407b4b5e4"],["E:/GitHubBlog/public/tags/python/index.html","00bc0a9c9a79ac50f3d7bf1073f810da"],["E:/GitHubBlog/public/tags/pytorch/index.html","cc920026cd7074dc666592aec9900a11"],["E:/GitHubBlog/public/tags/优化方法/index.html","0f573a32466f61162f49a3419d3323ee"],["E:/GitHubBlog/public/tags/总结/index.html","16710389a0b87f35e7870293bdb92f5a"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","c3b8febe665550f4330bd22824e1d4be"],["E:/GitHubBlog/public/tags/数据分析/index.html","06c540060b11a3024ce8042fa6c77dc0"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","0e47b23a544f54fe00814e87741f92a3"],["E:/GitHubBlog/public/tags/数据结构/index.html","e0c09f31ccf3ca4466e17fb1e3d93619"],["E:/GitHubBlog/public/tags/机器学习/index.html","528bd116b180144418982943fdd15daa"],["E:/GitHubBlog/public/tags/比赛/index.html","37933cec0a6bf004620cd87a66572df4"],["E:/GitHubBlog/public/tags/深度学习/index.html","1c76e90c62caefc35e2cbccc08a2c088"],["E:/GitHubBlog/public/tags/爬虫/index.html","8fee3298b5eecc303c4b041f40090fc5"],["E:/GitHubBlog/public/tags/笔记/index.html","bcd15a0d6be37d3f5f88e14fb59f4a3b"],["E:/GitHubBlog/public/tags/论文/index.html","3f928512f52a1531c8f6e457e20efb03"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","fb2ad4fdcd6b281e98bcda799cdac83a"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","5fe562fb58dec4661ba6989aa8aedd34"],["E:/GitHubBlog/public/tags/读书笔记/index.html","3e83f96cbe7357c5a88ddf4e0a35b358"]];
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







