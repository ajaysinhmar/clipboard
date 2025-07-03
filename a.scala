import io.gatling.core.Predef._
import io.gatling.http.Predef._
import scala.concurrent.duration._

class ApiPerformanceTest extends Simulation {

  val httpProtocol = http
    .baseUrl("https://api.example.com") // Replace with your base URL
    .acceptHeader("application/json")
    .contentTypeHeader("application/json")
    .authorizationHeader("Bearer your_bearer_token_here") // Replace with your token

  val scn = scenario("Authenticated API Load Test")
    .exec(
      http("Get Authenticated Resource")
        .get("/api/resource") // Replace with your endpoint
        .check(status.is(200))
    )

  setUp(
    scn.inject(
      rampUsers(100).during(30.seconds)
    )
  ).protocols(httpProtocol)
}
