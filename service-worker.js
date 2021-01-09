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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","e5f835aa8c504db57579bf625e208a9a"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","4405ac5391a4436351c08eb493a8548e"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","8a6d571b7783771d952298bc6b2ab42e"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","937f1f212e6d0a0ab1b3e84401452a54"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","f2339acd810469ceda70f038745b52af"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","6768ecdccdd0656975959326d269f7c9"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","a588f3568b6855d943b3d9e5be6497a4"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","6a9b366e2d9b628291b0fd6ea555abe2"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","037df5fe7406570c1de35a5fab58ea45"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","0eb72edeaffaf0f2e4e13d807fd03e71"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","2259f0bc056ba9cec1212fcf15d7788c"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","4361949cb3d0f8129a5367b064c51247"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","e239e9becedca26b4faa9e33e0fab6b9"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","43abc8930eb15f10cef06017d9150d73"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","c2c4da545017df27331c26e2c5529ed5"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","eb49b31ce1aae2e4bc4ca43892d43699"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","6b41dba9012ad2fc17b63cc05399c800"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","ee219ec65eb7bd062cedff1c6415cfd0"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","5026c5b70f12d21df256bbe79b14bd51"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","7b657e9eab3f9f7ed565e9bd716c3bdd"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","985e23bfd1f43a0f0cf80bbc7f9ff1d1"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","f0153d878e5a21d94e0c82dd39631a50"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","9b15090c83124fe3c62dcf16ef011adb"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","200065113286742e84fbb7ae4cc4c08c"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","fec44e8db2c67a8a411a69f1b27807e1"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","6f7849ef222e15b1ae90ade6a6e981a8"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","339b15489909d480880430bdaa46a432"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","aa23e886dfbb87915bb2c66eb939a282"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","767d7099752c834a69da361308854d26"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","5111a540a23ad248aba360b3889d18b8"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","64a0036df213edcae53dd49aa78f4902"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","35619bad9d36d85ea6fc287ca2b276c9"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","aa3f3bc595f8da746f9fd7ff6cd2a679"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","96133fe4d9901497024ce23912f0c44e"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","209bb473f209fc6e4992ca964eb09088"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","03a49571a90aa7f035bfe7497b2d2ba1"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","90a5f3c17c019a160115c67cc761073d"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","6652d2c6b861562cf0fd007a8d2b15d1"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","0a4ade83dc761113d4a0d00e8bf6ad89"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","a558e12ff4cb3d2b943b8bfb4bee312c"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","a7f02303ef3fc32d539974dc4deafda6"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","1853ff06e561418bbd13a4971ebd8e3b"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","8487a32ce266eba68d7296cbcc49b4a2"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","b057d668a72a40abc4fec7452fa86c21"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","f3beb427a9115ddff3300f0aee8b75da"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","f0dbad5f8a234ca9d6fbb515da2dd281"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","0551139a13a60ed464ff9e9c376f78db"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","fa8d87813d675cc71d63e627a83ff2a0"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","07d1d4aef11e0b6e5447c8e0296ec91a"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","66f7df3219488ceb587039f00ff96176"],["E:/GitHubBlog/public/archives/2020/01/index.html","3ca41942e23c90c271f3248ab3147433"],["E:/GitHubBlog/public/archives/2020/02/index.html","c8f5c7266ccb2efea2d27a7673a64cc1"],["E:/GitHubBlog/public/archives/2020/03/index.html","ab6e96d207505157f9b10571f16f98ae"],["E:/GitHubBlog/public/archives/2020/04/index.html","fece59640693595f9acd6cd39bdab6f7"],["E:/GitHubBlog/public/archives/2020/05/index.html","506007dd9561026e3d7a4d4acc7cd727"],["E:/GitHubBlog/public/archives/2020/07/index.html","4a3d84d469103ab206474ebe02ecc2bb"],["E:/GitHubBlog/public/archives/2020/08/index.html","ff41a10fb406609dcbf8e12ad20e4aba"],["E:/GitHubBlog/public/archives/2020/09/index.html","9b5f6b42489dc3aa9356c6a2bb7d1472"],["E:/GitHubBlog/public/archives/2020/10/index.html","f8c7b00c94b5b5486c08717d46b31836"],["E:/GitHubBlog/public/archives/2020/11/index.html","6c091836074153b1adee39877a4427fb"],["E:/GitHubBlog/public/archives/2020/12/index.html","e4cd1d825af5fcb90f0eb2f132c15526"],["E:/GitHubBlog/public/archives/2020/index.html","f940c7e7ca4f2885947bff210dc172e1"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","78511df9616933b270d09ecc761feb97"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","80a28f6a7f3d30dfd2f414a8b7a3292f"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","d7dcc3a513a785777fe9be861183c74c"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","4d8b4b77b8c271b5c7d8eeb346bf4c6c"],["E:/GitHubBlog/public/archives/2021/01/index.html","d4f2b39af66d0eb49bf189f5af5ddd31"],["E:/GitHubBlog/public/archives/2021/index.html","f14d2e672d6a043e4b3eb86385189eb5"],["E:/GitHubBlog/public/archives/index.html","1150bccd5d44f620af03e7a02b4f60ac"],["E:/GitHubBlog/public/archives/page/2/index.html","8b82dab14ea59efd62ac4adf1c93f44d"],["E:/GitHubBlog/public/archives/page/3/index.html","23f8ee64201b1a951a89983c03d3a4c6"],["E:/GitHubBlog/public/archives/page/4/index.html","b0679d65fe904fc0ea98800395bcf62f"],["E:/GitHubBlog/public/archives/page/5/index.html","90d341a19c5fe716d2b650ce1fff1158"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","a8453fb7509190680449cd5fbb460962"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","27cff5c25a536696cb7b05c3ed4ba275"],["E:/GitHubBlog/public/page/3/index.html","0509fe02eda04d0e686b1dc04c267160"],["E:/GitHubBlog/public/page/4/index.html","d468c278f714cac7f7b20ffb7776c0e8"],["E:/GitHubBlog/public/page/5/index.html","e0c8ba935ec76acff836f6a5b6eff56f"],["E:/GitHubBlog/public/tags/Android/index.html","0b002f0518b22122456dbe5a3687ca2b"],["E:/GitHubBlog/public/tags/NLP/index.html","e6d551b8b954f83a0a73ba75e1014a3a"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","ccd374624b2efba0b34177ae45b5bcd8"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","8def62760268b724b2eb2b216114cd99"],["E:/GitHubBlog/public/tags/R/index.html","e827172398324e3375c5ba560999d402"],["E:/GitHubBlog/public/tags/index.html","1a20d14881335fd822c4e85ab0a3ee6f"],["E:/GitHubBlog/public/tags/java/index.html","b1c55add19c3050afc248a31409965de"],["E:/GitHubBlog/public/tags/leetcode/index.html","af59d697ac046285ac1bcc0e514cdaee"],["E:/GitHubBlog/public/tags/python/index.html","334ece59e6ef045e0372fa7f7d22ba75"],["E:/GitHubBlog/public/tags/pytorch/index.html","a91b7b5d98fc10c39826a4006efc4a8a"],["E:/GitHubBlog/public/tags/优化方法/index.html","5b87a6c983d54cb4eecaf1aca8536262"],["E:/GitHubBlog/public/tags/总结/index.html","daabfe56e6665efa083b1c4a9e9fa9e9"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","25e5237330bcb26279a0aec08a6f1900"],["E:/GitHubBlog/public/tags/数据分析/index.html","114ef4fd43a3ca4946087016a89c516a"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","b6bd07e9790a7748b1f99b0298283e98"],["E:/GitHubBlog/public/tags/数据结构/index.html","682224692d6b28a27d8d0705ecb1facd"],["E:/GitHubBlog/public/tags/机器学习/index.html","077893549f7bc74011b490d55f75e5d5"],["E:/GitHubBlog/public/tags/深度学习/index.html","5eb495b8bcb6c371055ed5941c5273f5"],["E:/GitHubBlog/public/tags/爬虫/index.html","493b9d239e492ccad6f7e91d86abc9b1"],["E:/GitHubBlog/public/tags/笔记/index.html","01f706dc58e6fcddcc7ddec420f07cd1"],["E:/GitHubBlog/public/tags/论文/index.html","7bc3ae14f14809ccb0e31e3767a907d3"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","5fc5922edd36da778c25b57f85c7bb5c"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","40666fe2af3a9e8e2bb519ad965903bb"],["E:/GitHubBlog/public/tags/读书笔记/index.html","55d5f4a8a04cb964f62e15ce9078eea5"]];
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







