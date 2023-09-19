echo "$0"
echo "$(dirname $0)/.."
echo "$(realpath $0)"
echo "$(dirname $(realpath $0))/.."
PROJECT_HOME=$(cd "$(dirname $(realpath $0))/.."; pwd)
# PROJECT_HOME="$(cd "$(dirname $0)"/..; pwd)"
echo "Entered project home: ${PROJECT_HOME}"
