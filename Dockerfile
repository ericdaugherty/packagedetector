FROM golang:1.12 as builder

# Set the working directory outside $GOPATH to enable the support for modules.
WORKDIR /src

# Fetch dependencies first; they are less susceptible to change on every build
# and will therefore be cached for speeding up the next build
COPY ./go.mod ./go.sum ./
RUN go mod download

# Import the code from the context.
COPY ./ ./

# Build the executable to `/app`. Mark the build as statically linked.
RUN GOOS=linux CGO_ENABLED=0 go build \
    -installsuffix 'static' \
    -o /packagedetector .

FROM google/cloud-sdk:alpine
RUN apk add --no-cache \
    tzdata \
    ca-certificates
WORKDIR /root/
COPY --from=builder /packagedetector .
ENTRYPOINT [ "./packagedetector" ]
