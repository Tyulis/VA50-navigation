//   Copyright 2023 Grégori MIGNEROT, Élian BELMONTE, Benjamin STACH
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#include <map>
#include <vector>
#include <algorithm>

template<typename T>
std::vector<size_t> argsort(const std::vector<T> &array) {
    std::vector<size_t> indices(array.size());
    for (size_t i = 0; i < array.size(); i++)
        indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });

    return indices;
}