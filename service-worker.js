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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","088d3a29aeb9d88df81baf9fa6b3e166"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","4c42ab5747d3e84f1e8299d9f355ca58"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","302a69a8014b06c3c851d8624d14c9c5"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","5b31d652c7ccece2adf9537f7ffa1785"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","d07d1e8b01f011826c8962b20c0176a6"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","5dad6e39331995a7af1e7d777bd7e868"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","738d09c6ba53b55b667634b26059882b"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","9d460ef01c3d931330dfddf8c9c59d49"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","61d0d8ead76bd5229095531763943349"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","6f58aab09c1d7492367a793e138131f1"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","c545b289f2278be69b3f421d1d5c3cf8"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","9593dba122a6ce802ab069d1f4797a9b"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","f0c064bb8e7a42f2c6289ceea4acdf4b"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","83e58a01aed239dfc686b5dea09171b4"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","ecc162fbe6ec4f6a7fdf9d55620afa32"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","8ac5856bc1e1fb9923d8c8951c9eafb0"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","44839d7f3b51f5f97a0d86a1ac49050e"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","da72c29e534f8ed53c001bfdf424d22d"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","f8a9ed83a449bea82305bea00396c120"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","92eb263c41ff397d0293f632ce89735f"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","47d42e485db75b4f3afc184645341700"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","db94a20ee09505c8dd974d52770808bd"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","8de0eeb54ddefb13df6fddf6eda653d7"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","e0766a97a6b587cc417048f197598ac7"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","ac8578325169ccae585af6ae972b8d47"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","eba3fde35a2cb6e08cfc65649a67f106"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","e7716bc9ba3d72e4174f0d76b3d276e4"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","f5385dbca680c776f53c3291d98dc271"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","5145de88aec1e534e9252f634ce8c462"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","84f2cf93e09341a6a840bc0b1eeddfb9"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","919432ecf481ecf6ddb8de04fdd01afc"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","b470bd77dd60bd98c0b915eced4bc744"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","f4ae5074f065870153c6c2e22c28f7fa"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","d380d25faa5e6e0adc23b22e97639fc6"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","0f3ed7683115f0ff5bbdf8643c021cc1"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","29e67e7a9c79844d8f306430d6053477"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","e94bcc221aa4dc08a8c30dc12bed6e5c"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","7772c3ec21e873cee1efc5166b2c8b8e"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","8d8b661a6f6d3a7765ce9c2ac1304c6a"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","2e77bf15391442fb8590df6ae09a9bb1"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","c8cc60741dff1951436466f924ee81bc"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","799ea9bbbd1817936065712cf5a87996"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","9530e0694e329140e6e88fd705149679"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","ed28763c52e1be6f30f53842eff2b0a8"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","2068674938e04e035c30db61f69c7280"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","5779ec77c722c9de0d003de81d734ba2"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","26218d6343d38a12f48513af860a03db"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","ff7bbfba8d4a1d40f191ab59e24fe43c"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","0fa39710c5d41830c607d710bd68049c"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","cc6e2425b41a49062a3a0d3774f965df"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","9ede34776766f7953a0f9da7a6a41fb5"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","764cc8bfccc8794fe5104b81b1722b08"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","652bc49187cce4c9d86d9409792ea138"],["E:/GitHubBlog/public/archives/2020/01/index.html","8623e034d262f9fdf32f8d8b593a32f7"],["E:/GitHubBlog/public/archives/2020/02/index.html","40183d18dadb3470af1fc4bb67b08d75"],["E:/GitHubBlog/public/archives/2020/03/index.html","fe61831fbd49809a2a1f5f8714db5b1f"],["E:/GitHubBlog/public/archives/2020/04/index.html","cd7932f2e8d3ce6445e57d711aab45b2"],["E:/GitHubBlog/public/archives/2020/05/index.html","99af33d8a1bd798215c2eba59d89408a"],["E:/GitHubBlog/public/archives/2020/07/index.html","964afac662675bedae310e03e064b757"],["E:/GitHubBlog/public/archives/2020/08/index.html","9f5e2dd2334b518ff9452c7dac7b9cd5"],["E:/GitHubBlog/public/archives/2020/09/index.html","53fad30992954acc4d1b5f73b4429bef"],["E:/GitHubBlog/public/archives/2020/10/index.html","6dc88ded2c5d94cf22319393339c0f4c"],["E:/GitHubBlog/public/archives/2020/11/index.html","a7bb575f7354ef159b5802e7afd0d582"],["E:/GitHubBlog/public/archives/2020/12/index.html","ac6517d6a8792a2da2fbe77204696666"],["E:/GitHubBlog/public/archives/2020/index.html","fc416d130bfc77cd580ec380eb7c2e5b"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","551ad0cf7dcdb70205ee0da44d1668b3"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","a8777bde78f90f1f6a9ca5571f315aab"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","bb757bea4fda4466ee96933f52b01fb3"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","465a3124d6322ac34cfe2d1c5b33bf56"],["E:/GitHubBlog/public/archives/2021/01/index.html","04a31c48aaf10577bc7f863f16c6d41b"],["E:/GitHubBlog/public/archives/2021/index.html","3241c8ba6857bea8673a353d5f9d4428"],["E:/GitHubBlog/public/archives/index.html","26b21b62513b10efa3a0633d59131cd1"],["E:/GitHubBlog/public/archives/page/2/index.html","5e052652b63a40038265add1a7735286"],["E:/GitHubBlog/public/archives/page/3/index.html","bb869192b19ad0d587e5eaedbba0509f"],["E:/GitHubBlog/public/archives/page/4/index.html","ec1edfac82380049e9112fdaa898a0ff"],["E:/GitHubBlog/public/archives/page/5/index.html","4e5de9fcd4a00d1d8ccd9fa691c89b15"],["E:/GitHubBlog/public/archives/page/6/index.html","11b42e023176663741c3f33d30d4ccb3"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","01786e96b255599af7d26399939cff67"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","d85c3a8e1d57c1e57441fb39f3ccc947"],["E:/GitHubBlog/public/page/3/index.html","af7d8dd37ca3a3c931904f6b86995a9c"],["E:/GitHubBlog/public/page/4/index.html","ea09e60deadee093575671ad3f32e896"],["E:/GitHubBlog/public/page/5/index.html","939878108e8bc17c7e13de994a27bc2d"],["E:/GitHubBlog/public/page/6/index.html","d9c297a227b24b8af3b2a26b1e4ac5fb"],["E:/GitHubBlog/public/tags/Android/index.html","de739ee1682f763f778a60e2f2d86897"],["E:/GitHubBlog/public/tags/NLP/index.html","9b576c15d0d0fbd31d724282b6711778"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","70c0a04cfe30b851e12935360807cd53"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","3699a443067adb7c2ecd470193d42972"],["E:/GitHubBlog/public/tags/R/index.html","99d80a9dc2c7660b5957be8479459732"],["E:/GitHubBlog/public/tags/index.html","35582c1a80717982b6e661669593b9e1"],["E:/GitHubBlog/public/tags/java/index.html","5c6968db59ee16a4cfedc29782502db7"],["E:/GitHubBlog/public/tags/leetcode/index.html","61bc8ed93e9a38e661a2a394da48803e"],["E:/GitHubBlog/public/tags/python/index.html","95308ab2f60b254f65eec8ec5d8a4ac1"],["E:/GitHubBlog/public/tags/pytorch/index.html","b6f258d7d574f2283ba47dfd578f3941"],["E:/GitHubBlog/public/tags/优化方法/index.html","1ed92c89f28636e4b78a5b71a310def8"],["E:/GitHubBlog/public/tags/总结/index.html","bdaa615629896b66112b27580e51de24"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","a6d02b7b55a5366cbe0092c58e22a045"],["E:/GitHubBlog/public/tags/数据分析/index.html","5fa3ad20ee2426ae09b83ab953535fc7"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","1e37e005b127fab45909724ef066836a"],["E:/GitHubBlog/public/tags/数据结构/index.html","264ac01f5e26e97bd5bd249e2780e54c"],["E:/GitHubBlog/public/tags/机器学习/index.html","2729365bc0cb9aa32326d2bfcc0f91aa"],["E:/GitHubBlog/public/tags/深度学习/index.html","a8d9214cb7f707d016f4e0b6c362201d"],["E:/GitHubBlog/public/tags/爬虫/index.html","c83b71bd518fd510802b02f2cc0c4e0b"],["E:/GitHubBlog/public/tags/笔记/index.html","67f29fdf0ea5acda85a3327c60d8c3f8"],["E:/GitHubBlog/public/tags/论文/index.html","f44489c7d674cfae022f77cecb022396"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","a5f43b56fddef9d79da6526c954686ba"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","3a9601277d8641887e45f24d95215bc6"],["E:/GitHubBlog/public/tags/读书笔记/index.html","c2879e2a6c8090a3e99cbc10ed6728b4"]];
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







